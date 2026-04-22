function lossClassifierNet = build_lossClassifierNet(trained_net, T, C)
% =========================================================================
% Build a frozen loss network from the trained EEGNet classifier trunk
%
% Input:
%   trained_net : trained network returned by trainNetwork
%   T           : time dimension
%   C           : number of channels
%
% Output:
%   lossClassifierNet : dlnetwork used in generator training
%
% Notes:
%   1. This network starts from EEGNet's 'concat' output.
%   2. It reuses trained weights from the classifier part.
%   3. Dropout layers are removed for stable generator supervision.
%   4. classificationLayer is removed because dlnetwork does not need it.
%   5. Expected input format for forward():
%         [T 1 C B] with dlarray label 'SSCB'
% =========================================================================

    % Convert trained network to layerGraph
    lgraph_full = layerGraph(trained_net);
    allLayers = lgraph_full.Layers;
    layerNames = {allLayers.Name};

    % Find the split point
    concat_idx = find(strcmp(layerNames, 'concat'), 1);
    if isempty(concat_idx)
        error('Layer ''concat'' not found in trained_net.');
    end

    % Extract the classifier trunk after concat
    postLayers = allLayers(concat_idx+1:end);

    % Remove dropout + classification output layers for stable supervision
    keepMask = true(numel(postLayers), 1);
    for i = 1:numel(postLayers)
        if isa(postLayers(i), 'nnet.cnn.layer.DropoutLayer')
            keepMask(i) = false;
        end
        if isa(postLayers(i), 'nnet.cnn.layer.ClassificationOutputLayer')
            keepMask(i) = false;
        end
    end
    postLayers = postLayers(keepMask);

    % New input layer for F_feat / F_repaired
    featureInput = imageInputLayer([T 1 C], ...
        'Name', 'feature_input', ...
        'Normalization', 'none');

    % Build sequential classifier trunk
    classifierLayers = [
        featureInput
        postLayers
    ];

    % Convert to dlnetwork
    lossClassifierNet = dlnetwork(layerGraph(classifierLayers));

end
