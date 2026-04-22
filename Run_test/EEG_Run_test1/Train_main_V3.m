%% =========================================================
%  Train_Main_V3.m
%  EEG 通道级异常判定+重建系统 - MATLAB R2021a 可运行版（详细注释版）
%  功能：
%    1. EEGNet训练（通道独立卷积 + concat + 原EEGNet后续卷积分类）
%    2. 提取中间层特征 [N x T x C]，保证通道独立
%    3. Self-Attention矩阵（A_attention） 和 协方差矩阵（W_cov） 融合
%    4. 图构建
%    5. 简单判定
%    6. teacher+student知识蒸馏生成模型
%    
% =========================================================

clear; clc; close all;

restoredefaultpath
rehash toolboxcache

addpath('utils');   % 调用函数
addpath('models');  % 分类模型
addpath('data');    % 数据集
addpath('Train');   % 训练参数

fprintf('======== EEG 异常检测系统 训练开始 ========\n');

%% =========================================================
% 0. 读取数据
% =========================================================
load('train_data_subject 1_trialSplit.mat');;   % X_train [N,T,C_eeg], Y_train [N×1]
[N, T, C_eeg] = size(X_train);

fprintf('数据读取完成: N=%d T=%d C_eeg=%d\n', N, T, C_eeg);

%% =========================================================
% 1. 训练 EEGNet（通道独立卷积 + concat + 后续卷积分类）
% =========================================================
fprintf('Step 1: 训练 EEGNet...\n');

% 1.1 创建 EEGNet 网络
fullNet = EEGNetModel_ChannelIndependent(T, C_eeg);
fprintf('EEGNet 网络构建完成\n');
  
% MATLAB conv2d 输入要求: [H W C N]
% 将每个通道视为独立通道输入:
% 输入尺寸: [T 1 C N]
% MATLAB conv2d 输入要求: [H W C N]
% 前置通道独立卷积不使用池化，保持通道独立信息
X_cnn = permute(X_train, [2 3 1]); % [T C N]
X_cnn = reshape(X_cnn, [T 1 C_eeg N]); % [T 1 C N]
fprintf('训练数据准备完成，尺寸 [%d x %d x %d x %d]\n', size(X_cnn));

% 1.3 设置训练参数
options = trainingOptions('adam', ...
    'ExecutionEnvironment', 'cpu', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true);

fprintf('训练参数设置完成\n');

% 1.4 训练网络
trained_net = trainNetwork(X_cnn, categorical(Y_train), fullNet, options);
fprintf('EEGNet训练完成\n');

%% =========================================================
% 2. 提取中间层特征 [N x T x C]（通道独立特征）
% =========================================================
fprintf('Step 2: 提取中间层特征...\n');

feature_layer = 'concat'; % 通道独立卷积 concat 层

F_raw = activations(trained_net, X_cnn, feature_layer); % [T x 1 x C x N]
F_feat = permute(F_raw, [4 1 3 2]); % [N x T x C x 1]
F_feat = squeeze(F_feat);           % [N x T x C]
fprintf('修复模块输入特征提取完成\n');

fprintf('X_train size = [%d %d %d]\n', size(X_train,1), size(X_train,2), size(X_train,3));
fprintf('F_raw size   = [%d %d %d %d]\n', size(F_raw,1), size(F_raw,2), size(F_raw,3), size(F_raw,4));
fprintf('F_feat size  = [%d %d %d]\n', size(F_feat,1), size(F_feat,2), size(F_feat,3));

if size(F_feat, 2) ~= T
    error('Feature time dimension mismatch: X_train T=%d, F_feat T=%d.', T, size(F_feat, 2));
end

if size(F_feat, 3) ~= C_eeg
    error('Feature channel dimension mismatch: X_train C=%d, F_feat C=%d.', C_eeg, size(F_feat, 3));
end


%建立 classifier 子网络
net = trained_net;
lgraph = layerGraph(net);
concat_idx = find(strcmp({lgraph.Layers.Name}, 'concat'));% 找到 concat 层位
classifierLayers = lgraph.Layers(concat_idx+1:end);
featureInput = imageInputLayer([T 1 C_eeg], ...
    'Name','feature_input', ...
    'Normalization','none');  % 创建新输入层
classifierLayers = [
    featureInput
    classifierLayers
];
classifierNet = assembleNetwork(layerGraph(classifierLayers));
fprintf('classifierNet建立完成\n');

%建立 lossClassifierNet 子网络
lossClassifierNet = build_lossClassifierNet(trained_net, T, C_eeg);
fprintf('lossClassifierNet built from trained EEGNet\n');

% Sanity check: lossClassifierNet output size
B_test = min(8, size(F_feat,1));
X_test = btc2sscb(F_feat(1:B_test,:,:));     % [T 1 C B]
X_test = dlarray(single(X_test), 'SSCB');
Y_test = forward(lossClassifierNet, X_test);
disp('Sanity check: size of classifier output =');
disp(size(extractdata(Y_test)));   % expected: [4 B_test]

fprintf('CNN特征维度: N=%d D=%d C_feat=%d\n', size(F_feat));



%% =========================================================
% 3. 生成模型先验矩阵训练
% =========================================================
fprintf('Step 3: 训练生成模型先验矩阵...\n');

% 训练 Self-Attention 权重矩阵
fprintf('Self-Attention...\n');
A_feat = trainSelfAttention_Phys(F_feat, 30, 1e-3); % 30轮，学习率 1e-3
fprintf('Self-Attention 权重矩阵训练完成\n');

% 计算协方差矩阵 W_cov
fprintf('计算协方差矩阵 W_cov...\n');
W_cov = computeCovarianceMatrix(F_feat);  % 计算协方差矩阵
fprintf('协方差矩阵 W_cov 计算完成\n');

% 使用图学习优化邻接矩阵
fprintf('使用图学习优化邻接矩阵...\n');
alpha = 0.5;  % A_attention 和 W_cov权重融合系数
A_updated = trainGraphAttention(F_feat, A_feat, W_cov, 50, 1e-4, 1e-4, alpha);  % 50轮，学习率 1e-4
fprintf('图学习优化完成，邻接矩阵更新完成\n');

fprintf('Building signed repair adjacency matrix...\n');

A_signed_corr = computeSignedCorrelationA(F_feat);

% topK=3: best CE improvement in Diagnostic 4
% topK=0: best accuracy/RMSE in Diagnostic 4
repairTopK = 3;
A_repair = prepareAdjacencyForInterpolation(A_signed_corr, repairTopK);
fprintf('A_repair built from signed correlation matrix, topK=%d\n', repairTopK);
save('Train/A_signed_corr_1.mat', 'A_signed_corr');
save('Train/A_repair_1.mat', 'A_repair', 'repairTopK');

fprintf('生理能量模型训练完成\n');

%% =========================================================
% 4.5 新预处理数据下的敏感通道诊断
% =========================================================
fprintf('Step 4.5: Channel sensitivity diagnostics on current trialSplit data...\n');

runSensitivityDiagnostic = true;
runComboDiagnostic = true;

diagEvalNum = min(500, size(F_feat, 1));  % CPU友好；正式可改成 size(F_feat,1)
topMForCombo = 12;                        % K=2: 66组, K=3: 220组
comboKList = [2 3];

singleDiagFile = fullfile('Train', 'channel_sensitivity_single_channel.mat');
comboDiagFile  = fullfile('Train', 'channel_sensitivity_combo_K2K3.mat');

if runSensitivityDiagnostic
    channel_sensitivity = runSingleChannelSensitivityDiagnostic( ...
        F_feat, ...
        Y_train, ...
        lossClassifierNet, ...
        diagEvalNum, ...
        singleDiagFile);

    fprintf('Single-channel sensitivity diagnostic saved to:\n  %s\n', singleDiagFile);
else
    if exist(singleDiagFile, 'file')
        S_diag = load(singleDiagFile);
        channel_sensitivity = S_diag.channel_sensitivity;
        fprintf('Loaded existing single-channel sensitivity file:\n  %s\n', singleDiagFile);
    else
        error('Single-channel sensitivity file not found. Please run diagnostic first.');
    end
end

if runComboDiagnostic
    combo_sensitivity = runComboChannelSensitivityDiagnostic( ...
        F_feat, ...
        Y_train, ...
        lossClassifierNet, ...
        channel_sensitivity, ...
        comboKList, ...
        topMForCombo, ...
        diagEvalNum, ...
        comboDiagFile);

    fprintf('K=2/K=3 combo sensitivity diagnostic saved to:\n  %s\n', comboDiagFile);
end

%% =========================================================
% 5. 生成模型训练
% =========================================================
fprintf('Step 6: 训练生成模型...\n');

% 设置训练的参数
% 目标：
% 1. 保留 A_signed_corr / A_repair 的基础插值能力
% 2. 让 MLP 只学习 A-Interp 之后的判别性小修正
% 3. 降低 noisy label CE 对训练的主导，让 teacher logit/feature 蒸馏成为主驱动

max_epoch = 20;

% Route 1:
% A_signed_corr / A_repair 是主修复器；
% residual MLP 只作为安全蒸馏层。
lambda_logit = 5.0;
lambda_feat  = 1.0;
lambda_recon = 0.0;
lambda_ce    = 0.0;
learningRate = 5e-4;
K_max = 3;          
H = 256;
batchSize = 16;


[W1, b1, W2, b2, history] = Train_Generator( ...
    F_feat, ...
    A_repair, ...
    Y_train, ...
    lossClassifierNet, ...
    max_epoch, ...
    lambda_logit, ...
    lambda_feat, ...
    lambda_recon, ...
    lambda_ce, ...
    learningRate, ...
    K_max, ...
    H, ...
    batchSize);

fprintf('生成模型训练完成\n');

%% =========================================================
% Route 1 K=3 stable evaluation
% Clean / Bad / A-Interp / MLP-Repaired
fprintf('\n===== Route 1 K=3 Stable Evaluation =====\n');

route1EvalSets = {
    [27 55 2];
    [27 55 46];
    [27 55 37];
    [46 33 56];
    [37 33 56];
    [55 46 37]
};

% 推荐正式结果用全样本；如果想快速测试，可设为 false。
evalUseAllSamples = true;

% 当 evalUseAllSamples = false 时，重复随机抽样评估。
numEvalRepeats = 3;
evalNumPerRepeat = min(500, size(F_feat, 1));

numCases = numel(route1EvalSets);
methodNames = {'Clean', 'Bad', 'AInterp', 'MLP'};

if evalUseAllSamples
    numEvalRepeats = 1;
end

ce_all = nan(numCases, numEvalRepeats, 4);
acc_all = nan(numCases, numEvalRepeats, 4);
rmse_all = nan(numCases, numEvalRepeats, 4);

caseNames = cell(numCases, 1);

rng(2026);

for evalCase = 1:numCases

    testChannels = route1EvalSets{evalCase};
    testChannels = testChannels(:)';
    caseNames{evalCase} = mat2str(testChannels);

    for rr = 1:numEvalRepeats

        if evalUseAllSamples
            evalIdx = 1:size(F_feat, 1);
        else
            evalIdx = randperm(size(F_feat, 1), evalNumPerRepeat);
        end

        F_eval = single(F_feat(evalIdx, :, :));
        Y_eval = Y_train(evalIdx);

        if iscategorical(Y_eval)
            labels_eval = double(Y_eval);
        else
            labels_eval = double(Y_eval(:));
        end

        B_eval = size(F_eval, 1);
        C_eval = size(F_eval, 3);

        abnormal_mask_eval = zeros(B_eval, C_eval, 'single');
        abnormal_mask_eval(:, testChannels) = 1;

        F_bad = F_eval;
        F_bad(:, :, testChannels) = 0;

        F_interp = apply_A_interpolation_single_channel( ...
            F_bad, ...
            abnormal_mask_eval, ...
            A_repair);

        F_repaired = generate_channels( ...
            F_bad, ...
            abnormal_mask_eval, ...
            A_repair, ...
            W1, b1, W2, b2);

        if isa(F_repaired, 'dlarray')
            F_repaired_num = gather(extractdata(F_repaired));
        else
            F_repaired_num = F_repaired;
        end
        F_repaired_num = single(F_repaired_num);

        [ce_clean, acc_clean] = evalClassifierCEAcc(lossClassifierNet, F_eval, labels_eval);
        [ce_bad, acc_bad] = evalClassifierCEAcc(lossClassifierNet, F_bad, labels_eval);
        [ce_interp, acc_interp] = evalClassifierCEAcc(lossClassifierNet, F_interp, labels_eval);
        [ce_repair, acc_repair] = evalClassifierCEAcc(lossClassifierNet, F_repaired_num, labels_eval);

        true_ch = F_eval(:, :, testChannels);
        bad_ch = F_bad(:, :, testChannels);
        interp_ch = F_interp(:, :, testChannels);
        repair_ch = F_repaired_num(:, :, testChannels);

        rmse_clean = 0;
        rmse_bad = sqrt(mean((bad_ch(:) - true_ch(:)).^2));
        rmse_interp = sqrt(mean((interp_ch(:) - true_ch(:)).^2));
        rmse_repair = sqrt(mean((repair_ch(:) - true_ch(:)).^2));

        repair_interp_rmse = sqrt(mean((repair_ch(:) - interp_ch(:)).^2));
        relative_residual = repair_interp_rmse / (rmse_interp + 1e-8);

        ce_all(evalCase, rr, :) = [ce_clean, ce_bad, ce_interp, ce_repair];
        acc_all(evalCase, rr, :) = [acc_clean, acc_bad, acc_interp, acc_repair];
        rmse_all(evalCase, rr, :) = [rmse_clean, rmse_bad, rmse_interp, rmse_repair];

        fprintf('\nRoute 1 K=3 Evaluation, Channels %s, Repeat %d/%d:\n', ...
            mat2str(testChannels), rr, numEvalRepeats);

        fprintf('  Clean        CE = %.4f, Acc = %.4f, RMSE = %.4f\n', ...
            ce_clean, acc_clean, rmse_clean);
        fprintf('  Abnormal     CE = %.4f, Acc = %.4f, RMSE = %.4f\n', ...
            ce_bad, acc_bad, rmse_bad);
        fprintf('  A-Interp     CE = %.4f, Acc = %.4f, RMSE = %.4f\n', ...
            ce_interp, acc_interp, rmse_interp);
        fprintf('  MLP-Repaired CE = %.4f, Acc = %.4f, RMSE = %.4f\n', ...
            ce_repair, acc_repair, rmse_repair);

        fprintf('  MLP-Repaired vs A-Interp RMSE = %.6f\n', repair_interp_rmse);
        fprintf('  Relative residual size = %.6f\n', relative_residual);

        fprintf('  A-Interp - Bad         : dCE = %.4f, dAcc = %.4f, dRMSE = %.4f\n', ...
            ce_interp - ce_bad, acc_interp - acc_bad, rmse_interp - rmse_bad);
        fprintf('  MLP-Repaired - Bad     : dCE = %.4f, dAcc = %.4f, dRMSE = %.4f\n', ...
            ce_repair - ce_bad, acc_repair - acc_bad, rmse_repair - rmse_bad);
        fprintf('  MLP-Repaired - AInterp : dCE = %.4f, dAcc = %.4f, dRMSE = %.4f\n', ...
            ce_repair - ce_interp, acc_repair - acc_interp, rmse_repair - rmse_interp);
    end
end

% Summary across cases/repeats
meanCE = squeeze(mean(mean(ce_all, 1, 'omitnan'), 2, 'omitnan'));
meanAcc = squeeze(mean(mean(acc_all, 1, 'omitnan'), 2, 'omitnan'));
meanRMSE = squeeze(mean(mean(rmse_all, 1, 'omitnan'), 2, 'omitnan'));

stdCE = squeeze(std(reshape(ce_all, [], 4), 0, 1, 'omitnan'))';
stdAcc = squeeze(std(reshape(acc_all, [], 4), 0, 1, 'omitnan'))';
stdRMSE = squeeze(std(reshape(rmse_all, [], 4), 0, 1, 'omitnan'))';

summaryTable = table( ...
    methodNames(:), ...
    meanCE(:), ...
    stdCE(:), ...
    meanAcc(:), ...
    stdAcc(:), ...
    meanRMSE(:), ...
    stdRMSE(:), ...
    'VariableNames', { ...
        'Method', ...
        'MeanCE', ...
        'StdCE', ...
        'MeanAcc', ...
        'StdAcc', ...
        'MeanRMSE', ...
        'StdRMSE'});

disp(' ');
disp('===== Route 1 K=3 Summary Table =====');
disp(summaryTable);

badIdx = 2;
interpIdx = 3;
mlpIdx = 4;

fprintf('\n===== Route 1 K=3 Mean Improvement Summary =====\n');
fprintf('A-Interp - Bad      : dCE = %.4f, dAcc = %.4f, dRMSE = %.4f\n', ...
    meanCE(interpIdx) - meanCE(badIdx), ...
    meanAcc(interpIdx) - meanAcc(badIdx), ...
    meanRMSE(interpIdx) - meanRMSE(badIdx));

fprintf('MLP-Repaired - Bad  : dCE = %.4f, dAcc = %.4f, dRMSE = %.4f\n', ...
    meanCE(mlpIdx) - meanCE(badIdx), ...
    meanAcc(mlpIdx) - meanAcc(badIdx), ...
    meanRMSE(mlpIdx) - meanRMSE(badIdx));

fprintf('MLP-Repaired - AInterp: dCE = %.4f, dAcc = %.4f, dRMSE = %.4f\n', ...
    meanCE(mlpIdx) - meanCE(interpIdx), ...
    meanAcc(mlpIdx) - meanAcc(interpIdx), ...
    meanRMSE(mlpIdx) - meanRMSE(interpIdx));

% Save evaluation results
route1_k3_eval = struct();
route1_k3_eval.route = 'Route 1 stable K=3';
route1_k3_eval.evalUseAllSamples = evalUseAllSamples;
route1_k3_eval.numEvalRepeats = numEvalRepeats;
route1_k3_eval.evalNumPerRepeat = evalNumPerRepeat;
route1_k3_eval.evalSets = route1EvalSets;
route1_k3_eval.caseNames = caseNames;
route1_k3_eval.methodNames = methodNames;
route1_k3_eval.ce_all = ce_all;
route1_k3_eval.acc_all = acc_all;
route1_k3_eval.rmse_all = rmse_all;
route1_k3_eval.summaryTable = summaryTable;
route1_k3_eval.meanCE = meanCE;
route1_k3_eval.meanAcc = meanAcc;
route1_k3_eval.meanRMSE = meanRMSE;
route1_k3_eval.stdCE = stdCE;
route1_k3_eval.stdAcc = stdAcc;
route1_k3_eval.stdRMSE = stdRMSE;
route1_k3_eval.K_max = K_max;
route1_k3_eval.repairTopK = repairTopK;

if ~exist('Train', 'dir')
    mkdir('Train');
end

saveFile = fullfile('Train', 'route1_K3_stable_eval.mat');
save(saveFile, 'route1_k3_eval', '-v7.3');

fprintf('\nRoute 1 K=3 stable evaluation saved to:\n  %s\n', saveFile);

% Visualization
figure('Name', 'Route 1 K=3 Stable Evaluation Summary', 'Color', 'w');

subplot(1, 3, 1);
bar(meanCE);
set(gca, 'XTickLabel', methodNames);
ylabel('CE');
title('Classification CE');
grid on;

subplot(1, 3, 2);
bar(meanAcc);
set(gca, 'XTickLabel', methodNames);
ylabel('Accuracy');
title('Classification Accuracy');
ylim([0 1]);
grid on;

subplot(1, 3, 3);
bar(meanRMSE);
set(gca, 'XTickLabel', methodNames);
ylabel('RMSE');
title('Target-channel RMSE');
grid on;

%% =========================================================
% 6. 训练结果评估
% =========================================================
fprintf('评估生成模型\n');

% 1. 生成修复数据
fprintf('Phase 1: 开始生成修复结果...\n');

% 随机生成异常通道
abnormal_mask = zeros(C_eeg,1);
abnormal_channels = randperm(C_eeg, K_max);
abnormal_mask(abnormal_channels) = 1;

% 构造异常输入（用于测试）
F_abnormal = F_feat;
F_abnormal(:,:,abnormal_channels) = 0;

% 修复结果
F_repaired = generate_channels( ...
    F_abnormal, ...
    abnormal_mask, ...
    A_repair, ...
    W1, b1, W2, b2);

fprintf('修复完成\n');

% 2. 数值重建误差指标
fprintf('Phase 2: 计算重建误差...\n');

% RMSE
rmse_value = sqrt(mean((F_repaired(:)-F_feat(:)).^2));

% Relative Error
relative_error = norm(F_repaired(:)-F_feat(:)) / norm(F_feat(:));

fprintf('RMSE = %.6f\n', rmse_value);
fprintf('Relative Error = %.6f\n', relative_error);

% 可视化
figure;
bar([rmse_value, relative_error]);
xticklabels({'RMSE','Relative Error'});
title('Reconstruction Error Metrics');
ylabel('Value');
grid on;

% 3. 分类保持能力
fprintf('Phase 3: 计算分类恢复能力...\n');

% 分类前（异常输入）
pred_bad = classify(classifierNet, reshape(permute(F_abnormal,[2 4 3 1]),...
    [T 1 C_eeg size(F_abnormal,1)]));
acc_bad = mean(pred_bad == categorical(Y_train));

% 分类后（修复输入）
pred_repaired = classify(classifierNet, reshape(permute(F_repaired,[2 4 3 1]),...
    [T 1 C_eeg size(F_repaired,1)]));
acc_repaired = mean(pred_repaired == categorical(Y_train));

% Accuracy Recovery
acc_recovery = acc_repaired - acc_bad;

fprintf('Bad Accuracy = %.4f\n', acc_bad);
fprintf('Repaired Accuracy = %.4f\n', acc_repaired);
fprintf('Accuracy Recovery = %.4f\n', acc_recovery);

% 可视化
figure;
bar([acc_bad, acc_repaired, acc_recovery]);
xticklabels({'Bad','Repaired','Recovery'});
title('Classification Performance');
ylabel('Accuracy');
ylim([0 1]);
grid on;

% 4. 结构一致性指标
fprintf('Phase 4: 计算结构一致性...\n');

% 全局 Pearson
pearson_global = corr(F_repaired(:), F_feat(:));
fprintf('Global Pearson = %.4f\n', pearson_global);

% 通道级 Pearson
channel_corr = zeros(C_eeg,1);

for c = 1:C_eeg
    x1 = squeeze(F_repaired(:,:,c));
    x2 = squeeze(F_feat(:,:,c));
    
    channel_corr(c) = corr(x1(:), x2(:));
end

% 可视化
figure;
subplot(1,2,1)
bar(pearson_global)
title('Global Pearson')
ylim([0 1])

subplot(1,2,2)
bar(channel_corr)
title('Channel-wise Correlation')
xlabel('Channel')
ylabel('Correlation')
ylim([0 1])
grid on

% 5. 特征可视化对比
fprintf('Phase 5: 特征对比可视化...\n');

sample_id = 1;
figure;

subplot(1,3,1)
imagesc(squeeze(F_feat(sample_id,:,:)))
title('Original Feature')
xlabel('Channel')
ylabel('Time')

subplot(1,3,2)
imagesc(squeeze(F_abnormal(sample_id,:,:)))
title('Abnormal Input')
xlabel('Channel')
ylabel('Time')

subplot(1,3,3)
imagesc(squeeze(F_repaired(sample_id,:,:)))
title('Repaired Feature')
xlabel('Channel')
ylabel('Time')

sgtitle('Feature Comparison')
colormap jet
colorbar

% 可以绘制能量收敛曲线
figure;
plot(energy_history);
xlabel('Epoch');
ylabel('loss_history');
title('Energy Convergence History');


%% =========================================================
% 7. 关键模型和参数保存
% =========================================================
fprintf('Step 6: 保存模型和参数...\n');

% 分类模型冻结保存
save('Train/trained_net_1.mat', 'trained_net');
% 分类模型子网络保存
save('Train/classifierNet_1.mat','classifierNet');
% 生理能量模型保存
save('Train/A_feat_1.mat', 'A_feat');  % 保存 Self-Attention 权重矩阵
save('Train/W_cov_1.mat', 'W_cov');  % 保存 W_cov
save('Train/A_updated_1.mat', 'A_updated');  % 保存优化后的邻接矩阵
save('Train/energy_weights_1.mat', 'energy_weights');  % 保存能量权重
% 通道工程能量GMM分布模型保存
save('Train/gmm_models_1.mat','gmm_models');
% 能量融合模型保存
save('Train/fusion_params_1.mat', 'fusion_params');
% 生成融合模型保存
generator_model.A_repair = A_repair;
generator_model.A_signed_corr = A_signed_corr;
generator_model.repairTopK = repairTopK;

generator_model.W1 = W1;
generator_model.b1 = b1;
generator_model.W2 = W2;
generator_model.b2 = b2;

generator_model.K_max = K_max;
generator_model.H = H;
generator_model.batchSize = batchSize;

generator_model.lambda_logit = lambda_logit;
generator_model.lambda_feat = lambda_feat;
generator_model.lambda_recon = lambda_recon;
generator_model.lambda_ce = lambda_ce;
generator_model.learningRate = learningRate;
generator_model.max_epoch = max_epoch;

generator_model.feature_layer = feature_layer;
generator_model.input_format = 'F_feat [N T C], repaired by generate_channels';
generator_model.repair_rule = 'Route1: signed A-interp + conservative residual MLP';

save('Train/trained_generator_model_route1.mat', 'generator_model', 'history');

fprintf('模型和参数保存完成\n');

fprintf('======== 所有模型训练完成 ========\n');

%% =========================================================
% Self-Attention 权重训练函数
% =========================================================
function A = trainSelfAttention_Phys(F, numEpochs, lr)
    [N, D, C] = size(F);
    
    % 初始化权重矩阵A
    A = rand(C,C);
    A = (A + A') / 2; % 保证矩阵A对称
    
    alpha = 1e-3;  % Self-Attention的学习率
    lambda = 1e-4; % L2正则化系数（可以根据实际需求调整）
    
    % 开始训练
    for epoch = 1:numEpochs
        gradA = zeros(C,C); 
        loss = 0;
        
        % 遍历所有样本
        for i = 1:N
            X = squeeze(F(i,:,:))'; % [C x D]
            x_mean = mean(X,2);  % 计算每个样本的均值
            
            Dg = diag(sum(A,2));  % 计算度矩阵D
            L = Dg - A;            % 计算拉普拉斯矩阵L
            
            % 计算能量E，E作为生理能量的一个度量
            E = x_mean' * L * x_mean;
            loss = loss + E;  % 累加损失
            
            gradA = gradA - (x_mean * x_mean');  % 梯度计算
        end
        
        % 添加L2正则化项，约束A的大小
        gradA = gradA + 2 * lambda * A;
        
        % 更新权重矩阵A
        A = A - lr * gradA / N;  % 梯度下降更新
        A = (A + A') / 2;        % 保证矩阵A对称
        A(A < 0) = 0;            % 保证矩阵A的元素为非负
        
        % 打印训练过程中的损失值
        fprintf('Epoch %d Loss %.4f\n', epoch, loss / N);
    end
end

%% =========================================================
% 计算协方差矩阵函数
% =========================================================
function W_cov = computeCovarianceMatrix(F)
    % F: [N x T x C] 隐藏特征矩阵
    % 输出 W_cov: [C x C] 协方差矩阵
    
    [N, T, C] = size(F);  % 获取样本数 N，时间步长 T，通道数 C

    % 初始化协方差矩阵 W_cov
    W_cov = zeros(C, C);

    % 计算每一对通道之间的协方差
    for i = 1:C
        Xi = squeeze(F(:,:,i));
        Xi = Xi(:);

        for j = i:C

            Xj = squeeze(F(:,:,j));
            Xj = Xj(:);

            c = cov(Xi, Xj);
            val = c(1,2);

            W_cov(i,j) = val;
            W_cov(j,i) = val;   % 保证对称
        end
    end
    % 归一化
    W_cov = abs(W_cov);
    W_cov = W_cov / max(W_cov(:));   
end

%% =========================================================
% GAT图构建和图学习优化函数
% =========================================================
function [A_updated] = trainGraphAttention(F, A_attention, W_cov, numEpochs, lr, lambda_reg, alpha)
    % F: Hidden feature matrix, [N x T x C]
    % A_attention: Initial adjacency matrix (from Self-Attention)
    % W_cov: Covariance matrix (calculated from features)
    % numEpochs: Number of training epochs
    % lr: Learning rate
    % lambda_reg: Regularization factor for adjacency matrix A
    % alpha: Mixing coefficient for attention and covariance matrices (0 <= alpha <= 1)

    % Get the number of samples, time steps, and channels
    [N, T, C] = size(F);

    % Step 1: Combine A_attention and W_cov to get the final adjacency matrix W
    % Perform weighted sum of attention and covariance matrices
    W = alpha * A_attention + (1 - alpha) * W_cov;  % Weighted sum (α for attention, 1-α for covariance)

    A_updated = W;  % Initialize updated adjacency matrix with the combined W

    % Step 2: Construct Degree Matrix D and Laplacian L
    D = diag(sum(A_updated, 2));  % Degree matrix
    L = D - A_updated;            % Laplacian matrix

    % Step 3: Train Graph Attention (GAT)
    for epoch = 1:numEpochs
        gradA = zeros(C, C);  % Initialize gradient for A_updated
        loss = 0;

        % Iterate through each sample
        for i = 1:N  % For each sample
            X = squeeze(F(i, :, :))';  % Node features for the current sample
            x_mean = mean(X, 2);  % Compute mean feature per channel

            % Compute energy (loss)
            E_full = x_mean' * L * x_mean;
            loss = loss + E_full;

            % Gradient calculation for adjacency matrix A (Regularization)
            gradA = gradA + 2 * lambda_reg * (A_updated - eye(C));  % Regularization term
        end

        % Gradient update for adjacency matrix A
        A_updated = A_updated - lr * gradA / N;  % Update adjacency matrix
        A_updated = (A_updated + A_updated') / 2;  % Ensure symmetry

        % Print progress
        fprintf('Epoch %d Loss: %.4f\n', epoch, loss / N);
    end
    
    %归一化
    eps_val = 1e-8;
    row_sum = sum(A_updated,2);
    A_updated = A_updated ./ (row_sum + eps_val);
end

%% =========================================================
% Train_Generator - 生成训练函数，用于训练生成模型中MLP的参数
% ===========================================================
function [W1, b1, W2, b2, history] = Train_Generator( ...
    F_normal, ...
    A, ...
    Y_train, ...
    lossClassifierNet, ...
    max_epoch, ...
    lambda_logit, ...
    lambda_feat, ...
    lambda_recon, ...
    lambda_ce, ...
    learningRate, ...
    K_max, ...
    H, ...
    batchSize)

% Route 1 stable trainer.
%
% Goal:
%   A_repair / A-Interp is the main repair method.
%   Residual MLP is a conservative safety layer.
%
% This trainer avoids aggressive hard mining and avoids forcing MLP
% to overfit small residual gains.

N = size(F_normal, 1);
T = size(F_normal, 2);
C = size(F_normal, 3);
numClasses = 4;

classes = unique(Y_train(:))';
disp('Unique labels in Y_train:');
disp(classes);

if ~isequal(classes, 1:numClasses)
    error('Y_train must use labels 1..numClasses.');
end

if K_max < 1
    K_max = 1;
end

if K_max > 3
    warning('Route 1 stable trainer currently supports K=1, K=2, or K=3. K_max is forced to 3.');
    K_max = 3;
end

% ===== channel priority =====
channelPriority = 1:C;
diagFile = fullfile('Train', 'channel_sensitivity_single_channel.mat');

if exist(diagFile, 'file')
    S = load(diagFile);

    if isfield(S, 'channel_sensitivity')
        cs = S.channel_sensitivity;

        ceOrder = [];
        accOrder = [];

        if isfield(cs, 'ce_order')
            ceOrder = cs.ce_order(:)';
        end

        if isfield(cs, 'acc_order')
            accOrder = cs.acc_order(:)';
        end

        topCount = min(20, C);
        channelPriority = unique([ceOrder(1:min(topCount, numel(ceOrder))), ...
                                  accOrder(1:min(topCount, numel(accOrder)))], ...
                                  'stable');

        channelPriority = channelPriority(channelPriority >= 1 & channelPriority <= C);

        if isempty(channelPriority)
            channelPriority = 1:C;
        end
    end
end

comboFile = fullfile('Train', 'channel_sensitivity_combo_K2K3.mat');

if K_max == 1

    trainUnits = channelPriority(:);
    unitsPerEpoch = min(12, numel(trainUnits));

elseif K_max == 2

    if exist(comboFile, 'file')
        S_combo = load(comboFile);

        if isfield(S_combo, 'combo_sensitivity') && ...
                isfield(S_combo.combo_sensitivity, 'K2')

            K2info = S_combo.combo_sensitivity.K2;

            if isfield(K2info, 'order_by_dCE')
                trainUnits = K2info.combos(K2info.order_by_dCE, :);
            else
                trainUnits = K2info.combos;
            end

            trainUnits = trainUnits(all(trainUnits >= 1 & trainUnits <= C, 2), :);

            fprintf('Route 1 trainer loaded K=2 sensitive channel pairs from:\n  %s\n', comboFile);
        else
            trainUnits = buildRoute1Pairs(channelPriority, C);
            fprintf('K=2 combo file exists but field K2 is missing. Fallback to buildRoute1Pairs.\n');
        end
    else
        trainUnits = buildRoute1Pairs(channelPriority, C);
        fprintf('K=2 combo file not found. Fallback to buildRoute1Pairs.\n');
    end

    unitsPerEpoch = min(6, size(trainUnits, 1));

elseif K_max == 3

    if exist(comboFile, 'file')
        S_combo = load(comboFile);

        if isfield(S_combo, 'combo_sensitivity') && ...
                isfield(S_combo.combo_sensitivity, 'K3')

            K3info = S_combo.combo_sensitivity.K3;

            if isfield(K3info, 'order_by_dCE')
                trainUnits = K3info.combos(K3info.order_by_dCE, :);
            else
                trainUnits = K3info.combos;
            end

            trainUnits = trainUnits(all(trainUnits >= 1 & trainUnits <= C, 2), :);

            fprintf('Route 1 trainer loaded K=3 sensitive channel triples from:\n  %s\n', comboFile);
        else
            trainUnits = buildRoute1Triples(channelPriority, C);
            fprintf('K=3 combo file exists but field K3 is missing. Fallback to buildRoute1Triples.\n');
        end
    else
        trainUnits = buildRoute1Triples(channelPriority, C);
        fprintf('K=3 combo file not found. Fallback to buildRoute1Triples.\n');
    end

    unitsPerEpoch = min(6, size(trainUnits, 1));

end


batchesPerUnit = 2;
plotEvery = 5;

fprintf('Route 1 stable trainer: K=%d\n', K_max);
fprintf('Training units per epoch = %d, batches/unit = %d\n', unitsPerEpoch, batchesPerUnit);

% ===== history =====
history.total_loss = zeros(max_epoch, 1);
history.logit_loss = zeros(max_epoch, 1);
history.feature_loss = zeros(max_epoch, 1);
history.recon_abnormal_loss = zeros(max_epoch, 1);
history.label_ce_loss = zeros(max_epoch, 1);

history.recon_loss_channel = nan(max_epoch, C);
history.channel_visits = zeros(max_epoch, C);

history.k_used = K_max * ones(max_epoch, 1);
history.lr_used = learningRate * ones(max_epoch, 1);
history.stage = strings(max_epoch, 1);
history.trainUnits = trainUnits;

% ===== conservative residual initialization =====
W1 = dlarray(single(randn(T, H) * sqrt(2 / T)));
b1 = dlarray(single(zeros(1, H)));

% Exact zero residual at initialization:
% MLP-Repaired == A-Interp before training.
W2 = dlarray(single(zeros(H, T)));
b2 = dlarray(single(zeros(1, T)));

fprintf('Generator input feature size: N=%d, T=%d, C=%d\n', N, T, C);
fprintf('Generator W1 size: [%d %d]\n', size(W1,1), size(W1,2));
fprintf('Generator W2 size: [%d %d]\n', size(W2,1), size(W2,2));

A_dl = dlarray(single(A));

% ===== figure =====
fig = figure('Name', 'Route 1 Stable Generator Monitor', 'Color', 'w');
tiledlayout(fig, 2, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

ax1 = nexttile;
title(ax1, 'Loss Summary');
grid(ax1, 'on');
hold(ax1, 'on');
hTotal = plot(ax1, nan, nan, 'k-', 'LineWidth', 1.5);
hLogit = plot(ax1, nan, nan, 'r--', 'LineWidth', 1.2);
hFeat = plot(ax1, nan, nan, 'b-.', 'LineWidth', 1.2);
hRecon = plot(ax1, nan, nan, 'm:', 'LineWidth', 1.2);
hLCE = plot(ax1, nan, nan, 'g-', 'LineWidth', 1.0);
legend(ax1, {'Total','LogitDistill','Feature','Recon-Abn','LabelCE'}, 'Location', 'best');
xlabel(ax1, 'Epoch');
ylabel(ax1, 'Loss');

ax2 = nexttile;
title(ax2, 'Recon Loss per Channel');
xlabel(ax2, 'Epoch');
ylabel(ax2, 'Channel');

ax3 = nexttile;
title(ax3, 'Channel Visits');
xlabel(ax3, 'Epoch');
ylabel(ax3, 'Channel');

ax4 = nexttile;
title(ax4, 'Training Schedule');
grid(ax4, 'on');
hold(ax4, 'on');
hK = plot(ax4, nan, nan, 'm-', 'LineWidth', 1.5);
hLR = plot(ax4, nan, nan, 'c-.', 'LineWidth', 1.5);
legend(ax4, {'K','lr'}, 'Location', 'best');
xlabel(ax4, 'Epoch');
ylabel(ax4, 'Value');

ax5 = nexttile;
title(ax5, 'Mean Channel Recon');
xlabel(ax5, 'Channel');
ylabel(ax5, 'Recon Loss');
grid(ax5, 'on');

ax6 = nexttile;
title(ax6, 'Total Channel Visits');
xlabel(ax6, 'Channel');
ylabel(ax6, 'Visits');
grid(ax6, 'on');

for epoch = 1:max_epoch

    history.stage(epoch) = "Route1-stable";

    epoch_total = 0;
    epoch_logit = 0;
    epoch_feat = 0;
    epoch_recon = 0;
    epoch_lce = 0;
    epoch_steps = 0;

    epoch_recon_sum = zeros(C, 1);
    epoch_recon_count = zeros(C, 1);

    if K_max == 1
        unitIdx = mod((epoch - 1) * unitsPerEpoch + (1:unitsPerEpoch) - 1, ...
            numel(trainUnits)) + 1;
    else
        unitIdx = mod((epoch - 1) * unitsPerEpoch + (1:unitsPerEpoch) - 1, ...
            size(trainUnits, 1)) + 1;
    end

    for uu = 1:numel(unitIdx)

        if K_max == 1
            abnormal_channels = trainUnits(unitIdx(uu));
        else
            abnormal_channels = trainUnits(unitIdx(uu), :);
        end

        abnormal_channels = unique(abnormal_channels(:)');
        abnormal_channels = abnormal_channels(abnormal_channels >= 1 & abnormal_channels <= C);

        for bi = 1:batchesPerUnit

            B = min(batchSize, N);
            batchIdx = randperm(N, B);

            F_batch = single(F_normal(batchIdx, :, :));
            labels_batch = double(Y_train(batchIdx, :));

            abnormal_mask = zeros(C, 1, 'single');
            abnormal_mask(abnormal_channels) = 1;

            labels_one_hot = zeros(numClasses, B, 'single');
            for i = 1:B
                labels_one_hot(labels_batch(i), i) = 1;
            end
            labels_batch_dl = dlarray(labels_one_hot, 'CB');

            [loss, logit_loss, feature_loss, recon_abnormal_loss, label_ce_loss, recon_loss_channel, ...
                grad_W1, grad_b1, grad_W2, grad_b2] = dlfeval( ...
                @modelGradients, ...
                F_batch, ...
                dlarray(abnormal_mask), ...
                A_dl, ...
                W1, ...
                b1, ...
                W2, ...
                b2, ...
                lossClassifierNet, ...
                labels_batch_dl, ...
                dlarray(single(lambda_logit)), ...
                dlarray(single(lambda_feat)), ...
                dlarray(single(lambda_recon)), ...
                dlarray(single(lambda_ce)));

            W1 = W1 - learningRate .* grad_W1;
            b1 = b1 - learningRate .* grad_b1;
            W2 = W2 - learningRate .* grad_W2;
            b2 = b2 - learningRate .* grad_b2;

            epoch_total = epoch_total + double(gather(extractdata(loss)));
            epoch_logit = epoch_logit + double(gather(extractdata(logit_loss)));
            epoch_feat = epoch_feat + double(gather(extractdata(feature_loss)));
            epoch_recon = epoch_recon + double(gather(extractdata(recon_abnormal_loss)));
            epoch_lce = epoch_lce + double(gather(extractdata(label_ce_loss)));
            epoch_steps = epoch_steps + 1;

            recon_vec = double(gather(recon_loss_channel(:)));

            for kk = 1:numel(abnormal_channels)
                ch = abnormal_channels(kk);

                epoch_recon_sum(ch) = epoch_recon_sum(ch) + recon_vec(ch);
                epoch_recon_count(ch) = epoch_recon_count(ch) + 1;
                history.channel_visits(epoch, ch) = history.channel_visits(epoch, ch) + 1;
            end
        end
    end

    history.total_loss(epoch) = epoch_total / max(epoch_steps, 1);
    history.logit_loss(epoch) = epoch_logit / max(epoch_steps, 1);
    history.feature_loss(epoch) = epoch_feat / max(epoch_steps, 1);
    history.recon_abnormal_loss(epoch) = epoch_recon / max(epoch_steps, 1);
    history.label_ce_loss(epoch) = epoch_lce / max(epoch_steps, 1);

    valid = epoch_recon_count > 0;
    tmp = nan(C, 1);
    tmp(valid) = epoch_recon_sum(valid) ./ epoch_recon_count(valid);
    history.recon_loss_channel(epoch, :) = tmp';

    fprintf(['Epoch %d/%d [Route1 stable K=%d], Units=%d, Batches/Unit=%d, ' ...
             'Total=%.6f, Logit=%.6f, FEAT=%.6f, RECON=%.6f, LabelCE=%.6f\n'], ...
        epoch, max_epoch, K_max, unitsPerEpoch, batchesPerUnit, ...
        history.total_loss(epoch), history.logit_loss(epoch), ...
        history.feature_loss(epoch), history.recon_abnormal_loss(epoch), ...
        history.label_ce_loss(epoch));

    if mod(epoch, plotEvery) == 0 || epoch == 1 || epoch == max_epoch

        set(hTotal, 'XData', 1:epoch, 'YData', history.total_loss(1:epoch));
        set(hLogit, 'XData', 1:epoch, 'YData', history.logit_loss(1:epoch));
        set(hFeat, 'XData', 1:epoch, 'YData', history.feature_loss(1:epoch));
        set(hRecon, 'XData', 1:epoch, 'YData', history.recon_abnormal_loss(1:epoch));
        set(hLCE, 'XData', 1:epoch, 'YData', history.label_ce_loss(1:epoch));

        cla(ax2);
        imagesc(ax2, history.recon_loss_channel(1:epoch, :)');
        axis(ax2, 'xy');
        colorbar(ax2);
        xlabel(ax2, 'Epoch');
        ylabel(ax2, 'Channel');
        title(ax2, 'Recon Loss per Channel');

        cla(ax3);
        imagesc(ax3, history.channel_visits(1:epoch, :)');
        axis(ax3, 'xy');
        colorbar(ax3);
        xlabel(ax3, 'Epoch');
        ylabel(ax3, 'Channel');
        title(ax3, 'Channel Visits');

        set(hK, 'XData', 1:epoch, 'YData', history.k_used(1:epoch));
        set(hLR, 'XData', 1:epoch, 'YData', history.lr_used(1:epoch));

        cla(ax5);
        bar(ax5, mean(history.recon_loss_channel(1:epoch, :), 1, 'omitnan'));
        xlabel(ax5, 'Channel');
        ylabel(ax5, 'Recon Loss');
        title(ax5, 'Mean Channel Recon');
        grid(ax5, 'on');

        cla(ax6);
        bar(ax6, sum(history.channel_visits(1:epoch, :), 1));
        xlabel(ax6, 'Channel');
        ylabel(ax6, 'Visits');
        title(ax6, 'Total Channel Visits');
        grid(ax6, 'on');

        drawnow;
    end
end

W1 = extractdata(W1);
b1 = extractdata(b1);
W2 = extractdata(W2);
b2 = extractdata(b2);

end

%% 梯度更新
function [loss, logit_loss, feature_loss, recon_abnormal_loss, label_ce_loss, recon_loss_channel, ...
    grad_W1, grad_b1, grad_W2, grad_b2] = modelGradients( ...
    F_normal, ...
    abnormal_mask, ...
    A, ...
    W1, ...
    b1, ...
    W2, ...
    b2, ...
    lossClassifierNet, ...
    labels, ...
    lambda_logit, ...
    lambda_feat, ...
    lambda_recon, ...
    lambda_ce)

if isa(F_normal, 'dlarray')
    F_normal_num = single(extractdata(F_normal));
else
    F_normal_num = single(F_normal);
end

if isa(abnormal_mask, 'dlarray')
    abnormal_mask_num = single(extractdata(abnormal_mask));
else
    abnormal_mask_num = single(abnormal_mask);
end
abnormal_mask_num = abnormal_mask_num(:);

F_repaired = generate_channels( ...
    F_normal, ...
    abnormal_mask, ...
    A, ...
    W1, ...
    b1, ...
    W2, ...
    b2);

X_repair = btc2sscb(F_repaired);
X_normal = btc2sscb(F_normal_num);

if isa(X_repair, 'dlarray')
    X_repair = dlarray(stripdims(X_repair), 'SSCB');
else
    X_repair = dlarray(single(X_repair), 'SSCB');
end
X_normal = dlarray(single(X_normal), 'SSCB');

prob_repair = forward(lossClassifierNet, X_repair);
prob_clean  = forward(lossClassifierNet, X_normal);

feat_repair = forward(lossClassifierNet, X_repair, 'Outputs', 'fc1');
feat_clean  = forward(lossClassifierNet, X_normal, 'Outputs', 'fc1');

logit_repair = forward(lossClassifierNet, X_repair, 'Outputs', 'fc_out');
logit_clean  = forward(lossClassifierNet, X_normal, 'Outputs', 'fc_out');

logit_loss = mean((logit_repair - logit_clean).^2, 'all');
feature_loss = mean((feat_repair - feat_clean).^2, 'all');

abnormal_idx = find(abnormal_mask_num == 1);
if isempty(abnormal_idx)
    recon_abnormal_loss = dlarray(single(0));
else
    F_rep = stripdims(F_repaired);
    target_abn = single(F_normal_num(:,:,abnormal_idx));
    pred_abn = F_rep(:,:,abnormal_idx);
    denom = mean(target_abn.^2, 'all') + 1e-8;
    recon_abnormal_loss = mean((pred_abn - target_abn).^2, 'all') / denom;
end

label_ce_loss = crossentropy(prob_repair, labels);

loss = lambda_logit .* logit_loss + ...
       lambda_feat  .* feature_loss + ...
       lambda_recon .* recon_abnormal_loss + ...
       lambda_ce    .* label_ce_loss;

F_repaired_log = single(extractdata(F_repaired));
diff_sq = (F_repaired_log - F_normal_num).^2;
recon_loss_channel = squeeze(mean(mean(diff_sq, 1), 2));

[grad_W1, grad_b1, grad_W2, grad_b2] = dlgradient(loss, W1, b1, W2, b2);
end

%% 
function X = btc2sscb(F)
% Convert [B T C] to [T 1 C B]
X = permute(F, [2 4 3 1]);
end

%%
function pairList = buildRoute1Pairs(channelPriority, C)
% Build stable K=2 sensitive pairs for Route 1.

channelPriority = unique(channelPriority(:)', 'stable');
channelPriority = channelPriority(channelPriority >= 1 & channelPriority <= C);

if numel(channelPriority) < 2
    pairList = [];
    return;
end

anchor = channelPriority(1);
topChannels = channelPriority(1:min(12, numel(channelPriority)));

pairs = [];

for i = 2:numel(topChannels)
    pairs = [pairs; anchor, topChannels(i)]; %#ok<AGROW>
end

for i = 2:(numel(topChannels)-1)
    pairs = [pairs; topChannels(i), topChannels(i+1)]; %#ok<AGROW>
end

pairs = sort(pairs, 2);
pairList = unique(pairs, 'rows', 'stable');

end

%%
function tripleList = buildRoute1Triples(channelPriority, C)

channelPriority = channelPriority(:)';
channelPriority = channelPriority(channelPriority >= 1 & channelPriority <= C);
channelPriority = unique(channelPriority, 'stable');

if numel(channelPriority) < 3
    channelPriority = 1:C;
end

topCount = min(15, numel(channelPriority));
topChannels = channelPriority(1:topCount);

tripleList = nchoosek(topChannels, 3);

% Put triples containing the most sensitive channel earlier.
anchor = topChannels(1);
hasAnchor = any(tripleList == anchor, 2);
tripleList = [tripleList(hasAnchor, :); tripleList(~hasAnchor, :)];

end

%%
function A_out = prepareAdjacencyForInterpolation(A_in, topK)
% Prepare A for interpolation:
%   - remove self-connections
%   - optionally keep top-k absolute weights per row
%   - normalize by sum(abs(row)) to support signed matrices

A_out = single(A_in);
C = size(A_out, 1);

A_out(1:C+1:end) = 0;

if topK > 0 && topK < C
    A_sparse = zeros(size(A_out), 'single');

    for r = 1:C
        row = A_out(r, :);
        [~, order] = sort(abs(row), 'descend');
        keep = order(1:min(topK, C-1));
        A_sparse(r, keep) = row(keep);
    end

    A_out = A_sparse;
end

row_norm = sum(abs(A_out), 2) + 1e-8;
A_out = A_out ./ row_norm;

end

%%
function A_signed = computeSignedCorrelationA(F_feat)
% Compute signed channel correlation adjacency.
% F_feat: [N T C]

[N, T, C] = size(F_feat);

X = reshape(single(F_feat), [N*T, C]);
X = X - mean(X, 1);
X = X ./ (std(X, 0, 1) + 1e-6);

A_signed = corr(X);
A_signed(isnan(A_signed)) = 0;
A_signed = single(A_signed);

A_signed(1:C+1:end) = 0;

row_norm = sum(abs(A_signed), 2) + 1e-8;
A_signed = A_signed ./ row_norm;

end

%%
function F_interp = apply_A_interpolation_single_channel(F_bad, abnormal_mask, A)
% Apply A-only interpolation without MLP.
%
% F_bad         : [B T C]
% abnormal_mask : [C 1], [1 C], or [B C]
% A             : [C C], signed repair adjacency is allowed

[B, T, C] = size(F_bad);

F_interp = single(F_bad);
F_bad = single(F_bad);
A = single(A);

% Convert abnormal mask to channel-level mask.
abnormal_mask = single(abnormal_mask);

if isvector(abnormal_mask)
    abnormal_channel_mask = reshape(abnormal_mask, 1, []);
else
    abnormal_channel_mask = any(abnormal_mask == 1, 1);
    abnormal_channel_mask = single(abnormal_channel_mask);
end

if numel(abnormal_channel_mask) ~= C
    error('abnormal_mask must describe %d channels, but got %d elements.', ...
        C, numel(abnormal_channel_mask));
end

abnormal_channels = find(abnormal_channel_mask == 1);
if isempty(abnormal_channels)
    return;
end

normal_mask = (abnormal_channel_mask == 0);

A_dynamic = A;
A_dynamic(:, ~normal_mask) = 0;
A_dynamic(1:C+1:end) = 0;

% Signed matrix must use absolute row normalization.
row_norm = sum(abs(A_dynamic), 2) + 1e-8;
A_dynamic = A_dynamic ./ row_norm;

for ii = 1:numel(abnormal_channels)
    c = abnormal_channels(ii);

    a_row = reshape(A_dynamic(c, :), [1 1 C]);
    x_interp = squeeze(sum(F_bad .* a_row, 3));  % [B T]

    if B == 1
        x_interp = reshape(x_interp, [1 T]);
    end

    F_interp(:, :, c) = x_interp;
end

end

%%
function [ceValue, accValue] = evalClassifierCEAcc(lossClassifierNet, F_btc, labels)
% F_btc: [B T C]
% labels: [B 1] or [1 B], class index 1..K

X = dlarray(single(btc2sscb(F_btc)), 'SSCB');

prob = forward(lossClassifierNet, X);
probData = gather(extractdata(prob));   % [numClasses B]

labels = double(labels(:))';
B = numel(labels);

idx = sub2ind(size(probData), labels, 1:B);
ceValue = -mean(log(probData(idx) + 1e-8));

[~, pred] = max(probData, [], 1);
accValue = mean(double(pred) == labels);
end

%%
function pairList = buildSensitivePairs(channelPriority, C)
% Build K=2 sensitive channel pairs.
%
% The first sensitive channel is used as anchor.
% Then we add:
%   [anchor, other sensitive channels]
%   adjacent pairs among top sensitive channels

channelPriority = unique(channelPriority(:)', 'stable');
channelPriority = channelPriority(channelPriority >= 1 & channelPriority <= C);

if numel(channelPriority) < 2
    pairList = [];
    return;
end

anchor = channelPriority(1);
topChannels = channelPriority(1:min(12, numel(channelPriority)));

pairs = [];

% Anchor pairs, e.g. [46, 11], [46, 60], [46, 16].
for i = 2:numel(topChannels)
    pairs = [pairs; anchor, topChannels(i)]; %#ok<AGROW>
end

% Neighbor pairs within sensitive list, e.g. [11, 60], [60, 16].
for i = 2:(numel(topChannels)-1)
    pairs = [pairs; topChannels(i), topChannels(i+1)]; %#ok<AGROW>
end

% Add a few broader pairs to avoid overfitting only anchor pairs.
for i = 2:2:(numel(topChannels)-2)
    pairs = [pairs; topChannels(i), topChannels(i+2)]; %#ok<AGROW>
end

pairs = sort(pairs, 2);
pairList = unique(pairs, 'rows', 'stable');

end

%%
function channel_sensitivity = runSingleChannelSensitivityDiagnostic( ...
    F_feat, ...
    Y_train, ...
    lossClassifierNet, ...
    evalNum, ...
    saveFile)

fprintf('\n===== Diagnostic 1: Single-channel classification sensitivity =====\n');

N = size(F_feat, 1);
C = size(F_feat, 3);

evalNum = min(evalNum, N);
evalIdx = 1:evalNum;

F_eval = single(F_feat(evalIdx, :, :));
labels_eval = double(Y_train(evalIdx));
labels_eval = labels_eval(:);

[cleanCE, cleanAcc] = evalClassifierCEAcc(lossClassifierNet, F_eval, labels_eval);

badCE = zeros(C, 1);
badAcc = zeros(C, 1);
dCE = zeros(C, 1);
dAcc = zeros(C, 1);

fprintf('Clean baseline: CE = %.6f, Acc = %.4f\n', cleanCE, cleanAcc);

for ch = 1:C
    F_bad = F_eval;
    F_bad(:, :, ch) = 0;

    [badCE(ch), badAcc(ch)] = evalClassifierCEAcc(lossClassifierNet, F_bad, labels_eval);

    dCE(ch) = badCE(ch) - cleanCE;
    dAcc(ch) = cleanAcc - badAcc(ch);

    fprintf('Channel %02d: CE = %.6f, dCE = %.6f, Acc = %.4f, dAcc = %.4f\n', ...
        ch, badCE(ch), dCE(ch), badAcc(ch), dAcc(ch));
end

[~, ce_order] = sort(dCE, 'descend');
[~, acc_order] = sort(dAcc, 'descend');

channel_sensitivity = struct();
channel_sensitivity.cleanCE = cleanCE;
channel_sensitivity.cleanAcc = cleanAcc;
channel_sensitivity.badCE = badCE;
channel_sensitivity.badAcc = badAcc;
channel_sensitivity.dCE = dCE;
channel_sensitivity.dAcc = dAcc;
channel_sensitivity.ce_order = ce_order(:)';
channel_sensitivity.acc_order = acc_order(:)';
channel_sensitivity.evalNum = evalNum;
channel_sensitivity.note = 'Generated on current trialSplit preprocessing and current trained classifier.';

fprintf('\nTop channels by CE increase:\n');
topK = min(10, C);
for r = 1:topK
    ch = ce_order(r);
    fprintf('Rank %02d: Channel %02d, dCE = %.6f, BadCE = %.6f, BadAcc = %.4f\n', ...
        r, ch, dCE(ch), badCE(ch), badAcc(ch));
end

fprintf('\nTop channels by accuracy drop:\n');
for r = 1:topK
    ch = acc_order(r);
    fprintf('Rank %02d: Channel %02d, dAcc = %.4f, BadCE = %.6f, BadAcc = %.4f\n', ...
        r, ch, dAcc(ch), badCE(ch), badAcc(ch));
end

save(saveFile, 'channel_sensitivity');

figure('Name', 'Diagnostic 1: Single-channel Sensitivity', 'Color', 'w');

subplot(2, 2, 1);
bar(dCE);
xlabel('Channel');
ylabel('CE Increase');
title('CE Increase After Single-channel Zeroing');
grid on;

subplot(2, 2, 2);
bar(dAcc);
xlabel('Channel');
ylabel('Accuracy Drop');
title('Accuracy Drop After Single-channel Zeroing');
grid on;

subplot(2, 2, 3);
bar(ce_order(1:topK), dCE(ce_order(1:topK)));
xlabel('Channel');
ylabel('CE Increase');
title('Top Sensitive Channels by CE');
grid on;

subplot(2, 2, 4);
bar(acc_order(1:topK), dAcc(acc_order(1:topK)));
xlabel('Channel');
ylabel('Accuracy Drop');
title('Top Sensitive Channels by Accuracy');
grid on;

fprintf('Diagnostic 1 completed. Results saved to %s\n', saveFile);
fprintf('===============================================================\n');

end

%%
function combo_sensitivity = runComboChannelSensitivityDiagnostic( ...
    F_feat, ...
    Y_train, ...
    lossClassifierNet, ...
    channel_sensitivity, ...
    K_list, ...
    topM, ...
    evalNum, ...
    saveFile)

fprintf('\n===== Diagnostic 2: K=2/K=3 combo channel sensitivity =====\n');

N = size(F_feat, 1);
C = size(F_feat, 3);

evalNum = min(evalNum, N);
evalIdx = 1:evalNum;

F_eval = single(F_feat(evalIdx, :, :));
labels_eval = double(Y_train(evalIdx));
labels_eval = labels_eval(:);

[cleanCE, cleanAcc] = evalClassifierCEAcc(lossClassifierNet, F_eval, labels_eval);

ce_order = channel_sensitivity.ce_order(:)';
acc_order = channel_sensitivity.acc_order(:)';

topChannels = unique([ce_order(1:min(topM, numel(ce_order))), ...
                      acc_order(1:min(topM, numel(acc_order)))], ...
                      'stable');

topChannels = topChannels(topChannels >= 1 & topChannels <= C);

fprintf('Clean baseline: CE = %.6f, Acc = %.4f\n', cleanCE, cleanAcc);
fprintf('Top channels used for combo scan:\n');
disp(topChannels);

combo_sensitivity = struct();
combo_sensitivity.cleanCE = cleanCE;
combo_sensitivity.cleanAcc = cleanAcc;
combo_sensitivity.topChannels = topChannels;
combo_sensitivity.evalNum = evalNum;
combo_sensitivity.topM = topM;
combo_sensitivity.note = 'Combo sensitivity generated from current trialSplit preprocessing.';

for kk = 1:numel(K_list)

    K = K_list(kk);

    if numel(topChannels) < K
        warning('Not enough top channels for K=%d. Skipped.', K);
        continue;
    end

    combos = nchoosek(topChannels, K);
    numCombos = size(combos, 1);

    badCE = zeros(numCombos, 1);
    badAcc = zeros(numCombos, 1);
    dCE = zeros(numCombos, 1);
    dAcc = zeros(numCombos, 1);

    fprintf('\nScanning K=%d combos, total = %d\n', K, numCombos);

    for i = 1:numCombos
        channels = combos(i, :);

        F_bad = F_eval;
        F_bad(:, :, channels) = 0;

        [badCE(i), badAcc(i)] = evalClassifierCEAcc(lossClassifierNet, F_bad, labels_eval);

        dCE(i) = badCE(i) - cleanCE;
        dAcc(i) = cleanAcc - badAcc(i);

        fprintf('K=%d Combo %03d/%03d %s: CE = %.6f, dCE = %.6f, Acc = %.4f, dAcc = %.4f\n', ...
            K, i, numCombos, mat2str(channels), badCE(i), dCE(i), badAcc(i), dAcc(i));
    end

    [~, order_by_dCE] = sort(dCE, 'descend');
    [~, order_by_dAcc] = sort(dAcc, 'descend');

    fieldName = sprintf('K%d', K);

    combo_sensitivity.(fieldName).combos = combos;
    combo_sensitivity.(fieldName).badCE = badCE;
    combo_sensitivity.(fieldName).badAcc = badAcc;
    combo_sensitivity.(fieldName).dCE = dCE;
    combo_sensitivity.(fieldName).dAcc = dAcc;
    combo_sensitivity.(fieldName).order_by_dCE = order_by_dCE;
    combo_sensitivity.(fieldName).order_by_dAcc = order_by_dAcc;

    fprintf('\nTop K=%d combos by CE increase:\n', K);
    showTop = min(10, numCombos);
    for r = 1:showTop
        idx = order_by_dCE(r);
        fprintf('Rank %02d: Channels %s, dCE = %.6f, dAcc = %.4f, BadCE = %.6f, BadAcc = %.4f\n', ...
            r, mat2str(combos(idx, :)), dCE(idx), dAcc(idx), badCE(idx), badAcc(idx));
    end

    fprintf('\nTop K=%d combos by accuracy drop:\n', K);
    for r = 1:showTop
        idx = order_by_dAcc(r);
        fprintf('Rank %02d: Channels %s, dAcc = %.4f, dCE = %.6f, BadCE = %.6f, BadAcc = %.4f\n', ...
            r, mat2str(combos(idx, :)), dAcc(idx), dCE(idx), badCE(idx), badAcc(idx));
    end
end

save(saveFile, 'combo_sensitivity');

figure('Name', 'Diagnostic 2: Combo Sensitivity', 'Color', 'w');

plotIdx = 1;
for kk = 1:numel(K_list)
    K = K_list(kk);
    fieldName = sprintf('K%d', K);

    if ~isfield(combo_sensitivity, fieldName)
        continue;
    end

    info = combo_sensitivity.(fieldName);

    subplot(numel(K_list), 2, plotIdx);
    bar(info.dCE(info.order_by_dCE));
    xlabel('Combo rank');
    ylabel('CE Increase');
    title(sprintf('K=%d Combo CE Increase', K));
    grid on;
    plotIdx = plotIdx + 1;

    subplot(numel(K_list), 2, plotIdx);
    bar(info.dAcc(info.order_by_dAcc));
    xlabel('Combo rank');
    ylabel('Accuracy Drop');
    title(sprintf('K=%d Combo Accuracy Drop', K));
    grid on;
    plotIdx = plotIdx + 1;
end

fprintf('Diagnostic 2 completed. Results saved to %s\n', saveFile);
fprintf('===============================================================\n');

end
