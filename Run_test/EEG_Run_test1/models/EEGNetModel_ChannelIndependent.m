%% =========================================================
% EEGNetModel_ChannelIndependent 函数（通道独立卷积 + concat）
% =========================================================

function fullNet = EEGNetModel_ChannelIndependent(T, C)

numClasses = 4;

%% 输入层
inputLayer = imageInputLayer([T 1 C], ...
    'Name', 'input', ...
    'Normalization', 'none');

lgraph = layerGraph(inputLayer);

%% 通道独立卷积层
convLayerNames = cell(C,1);

for c = 1:C
    convName = ['conv_ch' num2str(c)];
    convLayerNames{c} = convName;

    convLayer = convolution2dLayer([3 1], 1, ...
        'Padding', 'same', ...
        'Name', convName);

    lgraph = addLayers(lgraph, convLayer);
    lgraph = connectLayers(lgraph, 'input', convName);
end

%% concat 嵌入层（修复模块输入位置）
concatLayer = concatenationLayer(3, C, 'Name', 'concat');
lgraph = addLayers(lgraph, concatLayer);

for c = 1:C
    lgraph = connectLayers(lgraph, ...
        convLayerNames{c}, ...
        ['concat/in' num2str(c)]);
end

%% 后续分类网络
postLayers = [
    batchNormalizationLayer('Name','bn1')
    eluLayer('Name','elu1')
    dropoutLayer(0.25,'Name','drop1')

    convolution2dLayer([3 3],16, ...
        'Padding','same', ...
        'Name','conv2')

    batchNormalizationLayer('Name','bn2')
    eluLayer('Name','elu2')
    dropoutLayer(0.25,'Name','drop2')

    convolution2dLayer([3 3],32, ...
        'Padding','same', ...
        'Name','conv3')

    batchNormalizationLayer('Name','bn3')
    eluLayer('Name','elu3')
    dropoutLayer(0.25,'Name','drop3')

    globalAveragePooling2dLayer('Name','global_pool')

    fullyConnectedLayer(64,'Name','fc1')
    reluLayer('Name','relu1')

    fullyConnectedLayer(numClasses,'Name','fc_out')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')
];

lgraph = addLayers(lgraph, postLayers);
lgraph = connectLayers(lgraph, 'concat', 'bn1');

%% 输出完整网络
fullNet = lgraph;

end