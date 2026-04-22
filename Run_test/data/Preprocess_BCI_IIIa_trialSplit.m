%% =========================================================
% Preprocess_BCI_IIIa_trialSplit.m
%
% BCI Competition IIIa preprocessing for channel-fault-tolerant
% EEG classification and repair experiments.
%
% Strategy:
%   1. Split train/test at trial level.
%   2. Slice train trials and test trials separately.
%   3. Normalize using train slices only.
%   4. Shuffle train/test slices separately.
%
% Output:
%   train_data_subject 1_trialSplit.mat
%   test_data_subject 1_trialSplit.mat
% =========================================================

clear; clc;

%% ==========================
% 1. Parameters
%% ==========================

subject_file = 'subject 1.mat';

train_ratio = 0.8;
rng_seed = 1;

% Motor imagery window relative to event onset.
t_start = 0.5;
t_end   = 3.0;

% Sliding window.
% If fs = 250:
%   slice_sec = 1.0 -> 250 samples
%   step_sec  = 0.2 -> 50 samples
slice_sec = 1.0;
step_sec  = 0.2;

normalize_flag = true;

%% ==========================
% 2. Load Data
%% ==========================

load(subject_file);  % Expected variables: HDR, band or s

fs = HDR.SampleRate;

if exist('band', 'var')
    EEG = band;
elseif exist('s', 'var')
    EEG = s;
else
    error('Cannot find EEG data variable. Expected variable band or s.');
end

EEG = single(EEG);

slice_length = round(slice_sec * fs);
step_size = round(step_sec * fs);

if slice_length <= 0 || step_size <= 0
    error('Invalid slice_length or step_size. Check fs, slice_sec, and step_sec.');
end

fprintf('Sampling rate: %d Hz\n', fs);
fprintf('Raw EEG size : [%d x %d]\n', size(EEG, 1), size(EEG, 2));
fprintf('Slice length : %d samples, %.3f s\n', slice_length, slice_length / fs);
fprintf('Step size    : %d samples, %.3f s\n', step_size, step_size / fs);

%% ==========================
% 3. Find MI Events
%% ==========================

event_types = HDR.EVENT.TYP;
event_pos   = HDR.EVENT.POS;

MI_codes = [769 770 771 772];

trial_mask = ismember(event_types, MI_codes);

MI_pos   = event_pos(trial_mask);
MI_types = event_types(trial_mask);
MI_labels = MI_types - 768;  % 1,2,3,4

num_trials = numel(MI_pos);

fprintf('Original MI trials: %d\n', num_trials);

if num_trials < 2
    error('Too few MI trials.');
end

%% ==========================
% 4. Stratified Trial-Level Split
%% ==========================

rng(rng_seed);

train_trial_idx = [];
test_trial_idx = [];

for cls = 1:numel(MI_codes)
    class_code = MI_codes(cls);
    idx_cls = find(MI_types == class_code);
    idx_cls = idx_cls(randperm(numel(idx_cls)));

    n_train_cls = round(train_ratio * numel(idx_cls));

    train_trial_idx = [train_trial_idx; idx_cls(1:n_train_cls)];
    test_trial_idx  = [test_trial_idx; idx_cls(n_train_cls + 1:end)];
end

train_trial_idx = train_trial_idx(randperm(numel(train_trial_idx)));
test_trial_idx  = test_trial_idx(randperm(numel(test_trial_idx)));

fprintf('Train trials: %d\n', numel(train_trial_idx));
fprintf('Test trials : %d\n', numel(test_trial_idx));

fprintf('Train trial label distribution:\n');
print_label_count(MI_labels(train_trial_idx));

fprintf('Test trial label distribution:\n');
print_label_count(MI_labels(test_trial_idx));

%% ==========================
% 5. Slice Train/Test Trials Separately
%% ==========================

[X_train, Y_train, train_slice_info] = slice_trials_from_events( ...
    EEG, ...
    MI_pos(train_trial_idx), ...
    MI_types(train_trial_idx), ...
    train_trial_idx, ...
    fs, ...
    t_start, ...
    t_end, ...
    slice_length, ...
    step_size);

[X_test, Y_test, test_slice_info] = slice_trials_from_events( ...
    EEG, ...
    MI_pos(test_trial_idx), ...
    MI_types(test_trial_idx), ...
    test_trial_idx, ...
    fs, ...
    t_start, ...
    t_end, ...
    slice_length, ...
    step_size);

fprintf('Train slices before cleaning: %d\n', size(X_train, 1));
fprintf('Test slices before cleaning : %d\n', size(X_test, 1));

%% ==========================
% 6. Remove Invalid Samples
%% ==========================

[X_train, Y_train, train_slice_info] = remove_invalid_slices( ...
    X_train, Y_train, train_slice_info);

[X_test, Y_test, test_slice_info] = remove_invalid_slices( ...
    X_test, Y_test, test_slice_info);

fprintf('Train slices after cleaning: %d\n', size(X_train, 1));
fprintf('Test slices after cleaning : %d\n', size(X_test, 1));

%% ==========================
% 7. Normalize Using Train Statistics Only
%% ==========================

norm_params = struct();

if normalize_flag
    C = size(X_train, 3);

    X_train_2d = reshape(X_train, [], C);

    mu = mean(X_train_2d, 1);
    sigma = std(X_train_2d, 0, 1);
    sigma(sigma == 0) = 1;

    mu = reshape(single(mu), [1 1 C]);
    sigma = reshape(single(sigma), [1 1 C]);

    X_train = (X_train - mu) ./ sigma;
    X_test  = (X_test  - mu) ./ sigma;

    norm_params.mu = mu;
    norm_params.sigma = sigma;
    norm_params.mode = 'channel-wise, train slices only';
else
    norm_params.mu = [];
    norm_params.sigma = [];
    norm_params.mode = 'none';
end

%% ==========================
% 8. Shuffle Train/Test Slices Separately
%% ==========================

rng(rng_seed);

perm_train = randperm(size(X_train, 1));
X_train = X_train(perm_train, :, :);
Y_train = Y_train(perm_train);
train_slice_info = train_slice_info(perm_train);

perm_test = randperm(size(X_test, 1));
X_test = X_test(perm_test, :, :);
Y_test = Y_test(perm_test);
test_slice_info = test_slice_info(perm_test);

%% ==========================
% 9. Summary
%% ==========================

fprintf('\n===== Preprocessing Summary =====\n');
fprintf('Subject file       : %s\n', subject_file);
fprintf('Sampling rate      : %d Hz\n', fs);
fprintf('Time window        : %.2f s to %.2f s\n', t_start, t_end);
fprintf('Slice length       : %d samples, %.3f s\n', slice_length, slice_length / fs);
fprintf('Step size          : %d samples, %.3f s\n', step_size, step_size / fs);
fprintf('Train trial count  : %d\n', numel(train_trial_idx));
fprintf('Test trial count   : %d\n', numel(test_trial_idx));
fprintf('Train slice count  : %d\n', size(X_train, 1));
fprintf('Test slice count   : %d\n', size(X_test, 1));
fprintf('Channels           : %d\n', size(X_train, 3));
fprintf('Normalization      : %s\n', norm_params.mode);

fprintf('Train slice label distribution:\n');
print_label_count(Y_train);

fprintf('Test slice label distribution:\n');
print_label_count(Y_test);

%% ==========================
% 10. Save Files
%% ==========================

subject_name = erase(subject_file, '.mat');

preprocess_params = struct();
preprocess_params.subject_file = subject_file;
preprocess_params.train_ratio = train_ratio;
preprocess_params.rng_seed = rng_seed;
preprocess_params.t_start = t_start;
preprocess_params.t_end = t_end;
preprocess_params.slice_sec = slice_sec;
preprocess_params.step_sec = step_sec;
preprocess_params.slice_length = slice_length;
preprocess_params.step_size = step_size;
preprocess_params.normalize_flag = normalize_flag;
preprocess_params.fs = fs;
preprocess_params.MI_codes = MI_codes;
preprocess_params.train_trial_idx = train_trial_idx;
preprocess_params.test_trial_idx = test_trial_idx;

train_file = ['train_data_' subject_name '_trialSplit.mat'];
test_file  = ['test_data_' subject_name '_trialSplit.mat'];

save(train_file, ...
    'X_train', ...
    'Y_train', ...
    'train_slice_info', ...
    'preprocess_params', ...
    'norm_params', ...
    '-v7.3');

save(test_file, ...
    'X_test', ...
    'Y_test', ...
    'test_slice_info', ...
    'preprocess_params', ...
    'norm_params', ...
    '-v7.3');

fprintf('\nSaved:\n');
fprintf('  %s\n', train_file);
fprintf('  %s\n', test_file);
fprintf('==============================\n');

%% =========================================================
% Local Functions
%% =========================================================

function [X, Y, slice_info] = slice_trials_from_events( ...
    EEG, ...
    MI_pos, ...
    MI_types, ...
    original_trial_idx, ...
    fs, ...
    t_start, ...
    t_end, ...
    slice_length, ...
    step_size)

num_trials = numel(MI_pos);
C = size(EEG, 2);

window_samples = round((t_end - t_start) * fs);
max_slices_per_trial = floor((window_samples - slice_length) / step_size) + 1;
max_slices_per_trial = max(max_slices_per_trial, 0);

max_total_slices = max_slices_per_trial * num_trials;

if max_total_slices <= 0
    error('No valid slices. Check t_start, t_end, slice_length, and step_size.');
end

X = zeros(max_total_slices, slice_length, C, 'single');
Y = zeros(max_total_slices, 1);

slice_info = repmat(struct( ...
    'original_trial_idx', [], ...
    'event_type', [], ...
    'label', [], ...
    'start_sample', [], ...
    'end_sample', [], ...
    'slice_order_in_trial', []), ...
    max_total_slices, 1);

slice_count = 0;

for i = 1:num_trials
    event_start = MI_pos(i);

    window_start = event_start + round(t_start * fs);
    window_end   = event_start + round(t_end * fs);

    last_start = window_end - slice_length + 1;

    if last_start < window_start
        warning('Trial %d has no valid slice. Skipped.', original_trial_idx(i));
        continue;
    end

    slice_order = 0;

    for start_sample = window_start:step_size:last_start
        end_sample = start_sample + slice_length - 1;

        if start_sample < 1 || end_sample > size(EEG, 1)
            continue;
        end

        segment = EEG(start_sample:end_sample, :);

        if any(isnan(segment(:))) || any(isinf(segment(:)))
            continue;
        end

        slice_count = slice_count + 1;
        slice_order = slice_order + 1;

        X(slice_count, :, :) = single(segment);
        Y(slice_count) = MI_types(i) - 768;

        slice_info(slice_count).original_trial_idx = original_trial_idx(i);
        slice_info(slice_count).event_type = MI_types(i);
        slice_info(slice_count).label = MI_types(i) - 768;
        slice_info(slice_count).start_sample = start_sample;
        slice_info(slice_count).end_sample = end_sample;
        slice_info(slice_count).slice_order_in_trial = slice_order;
    end
end

X = X(1:slice_count, :, :);
Y = Y(1:slice_count);
slice_info = slice_info(1:slice_count);

end

function [X, Y, slice_info] = remove_invalid_slices(X, Y, slice_info)

if isempty(X)
    return;
end

energy = squeeze(sum(sum(abs(X), 2), 3));
valid_idx = energy ~= 0;

X = X(valid_idx, :, :);
Y = Y(valid_idx);
slice_info = slice_info(valid_idx);

end

function print_label_count(Y)

classes = unique(Y(:))';

for c = classes
    fprintf('  Class %d: %d\n', c, sum(Y(:) == c));
end

end
