function F_out = generate_channels(F_in, abnormal_mask, A, W1, b1, W2, b2, residualScale)
% =========================================================================
% Route 1 stable residual repair module
%
% Main repair:
%   x_base = signed A-interpolation
%
% Residual safety layer:
%   repaired = x_base + residualScale * MLP(x_base)
%
% If W2 = 0 and b2 = 0, MLP-Repaired is exactly A-Interp.
% =========================================================================

    if nargin < 8 || isempty(residualScale)
        residualScale = single(0.05);
    end
    residualScale = single(residualScale);

    if ~isa(F_in, 'dlarray')
        F_base = dlarray(single(F_in));
    else
        F_base = F_in;
    end

    [B, T, C] = size(F_base);
    F_out = F_base;

    if isa(abnormal_mask, 'dlarray')
        abnormal_mask_num = extractdata(abnormal_mask);
    else
        abnormal_mask_num = abnormal_mask;
    end
    abnormal_mask_num = single(abnormal_mask_num);

    if isvector(abnormal_mask_num)
        abnormal_channel_mask = reshape(abnormal_mask_num, 1, []);
    else
        abnormal_channel_mask = single(any(abnormal_mask_num == 1, 1));
    end

    if numel(abnormal_channel_mask) ~= C
        error('abnormal_mask must describe %d channels.', C);
    end

    abnormal_channels = find(abnormal_channel_mask == 1);
    if isempty(abnormal_channels)
        return;
    end

    if isa(A, 'dlarray')
        A_num = extractdata(A);
    else
        A_num = A;
    end
    A_num = single(A_num);

    normal_mask = abnormal_channel_mask == 0;

    A_dynamic = A_num;
    A_dynamic(:, ~normal_mask) = 0;
    A_dynamic(1:C+1:end) = 0;

    % Signed A must use absolute row normalization.
    row_norm = sum(abs(A_dynamic), 2) + 1e-8;
    A_dynamic = A_dynamic ./ row_norm;

    F_num = single(extractdata(F_base));

    for c = abnormal_channels

        a_row = reshape(A_dynamic(c, :), [1 1 C]);
        x_base_num = squeeze(sum(F_num .* a_row, 3));

        if B == 1
            x_base_num = reshape(x_base_num, [1 T]);
        end

        x_base = dlarray(single(x_base_num));

        xT = size(x_base, 2);
        wT = size(W1, 1);
        wH = size(W1, 2);

        if xT ~= wT
            error(['generate_channels dimension mismatch before W1 multiplication: ', ...
                   'x_base is [%d x %d], W1 is [%d x %d]. ', ...
                   'Expected size(x_base,2) == size(W1,1). ', ...
                   'Check whether F_in is [B T C] and whether W1 was initialized with the same T.'], ...
                   size(x_base, 1), size(x_base, 2), size(W1, 1), size(W1, 2));
        end

        if size(b1, 2) ~= wH
            error('b1 dimension mismatch: b1 is [%d x %d], expected [1 x %d].', ...
                  size(b1, 1), size(b1, 2), wH);
        end

        if size(W2, 1) ~= wH
            error('W2 dimension mismatch: W2 is [%d x %d], expected first dimension %d.', ...
                  size(W2, 1), size(W2, 2), wH);
        end

        if size(W2, 2) ~= xT
            error('W2 dimension mismatch: W2 is [%d x %d], expected second dimension %d.', ...
                  size(W2, 1), size(W2, 2), xT);
        end

        h1 = x_base * W1 + b1;
        h1 = max(h1, 0);

        residual = h1 * W2 + b2;
        repaired = x_base + residualScale * residual;

        F_out(:, :, c) = repaired;
    end
end
