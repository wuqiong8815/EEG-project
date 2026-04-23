function F_out = generate_channels(F_in, abnormal_mask, A, varargin)
% =========================================================================
% generate_channels  -  通道修复模块
%
% 修复公式（新版）：
%   x_base_c  = Σ_j  A_dyn[c, j] * F_in[:, :, j]               % signed A-Interp
%   z_c       = [ x_base_c , ctx(obs) , E_ch(:, c) , E_mask * m ]
%   h         = ReLU( z_c * W1 + b1 )
%   delta_c   = h * W2 + b2
%   F_out[c]  = x_base_c + residualScale .* delta_c
%
% 与旧版的关键差异：
%   1. 不再用 extractdata 把 x_base 切成普通数组，dlarray 全程保持，
%      梯度可完整回传到 MLP 及后续可学习参数。
%   2. MLP 输入从 [B x T] 扩展为 [B x (T + T*useContext + E_ch + E_mask)]，
%      新增三类条件信号：
%        - 观测通道时间池化上下文 ctx  : [B x T]         (useContext=true 时拼接)
%        - 通道 one-hot embedding E_ch : [B x E_ch]      (传入 E_ch 时拼接)
%        - mask embedding E_mask*m     : [B x E_mask]    (传入 E_mask 时拼接)
%   3. residualScale 默认 0.2（旧版 0.05 过小导致 MLP 梯度几乎为零），
%      也可作为 dlarray 参数传入，由训练器一起学习。
%   4. 动态邻接矩阵 A_dyn 按当前 mask 关闭所有缺失通道的借用，再做
%      sum(|row|) 归一化，兼容 signed 邻接。
%
% 调用方式：
%   向后兼容（旧接口）
%       F_out = generate_channels(F_in, mask, A, W1, b1, W2, b2);
%       F_out = generate_channels(F_in, mask, A, W1, b1, W2, b2, residualScale);
%
%   新接口（推荐，条件化 MLP）：
%       gen_params.W1, .b1, .W2, .b2        (必填)
%       gen_params.E_ch          (可选, [E_ch_dim  x C])
%       gen_params.E_mask        (可选, [E_mask_dim x C])
%       gen_params.residualScale (可选, 标量或 dlarray 标量)
%       gen_params.useContext    (可选, logical, 默认 false)
%       F_out = generate_channels(F_in, mask, A, gen_params);
%
% 输入：
%   F_in          : [B x T x C]  或 dlarray，观测 / 被置零后的特征
%   abnormal_mask : [C x 1]、[1 x C] 或 [B x C]，1 = 该通道缺失
%   A             : [C x C] 邻接矩阵（允许 signed）
%
% 输出：
%   F_out         : [B x T x C] dlarray（若 F_in 为 dlarray，保持一致）
% =========================================================================

    % ---------- 1. 参数解析：兼容老接口 / 新接口 ----------
    if numel(varargin) == 1 && isstruct(varargin{1})
        gp = varargin{1};
        if ~all(isfield(gp, {'W1', 'b1', 'W2', 'b2'}))
            error('generate_channels: gen_params must contain W1, b1, W2, b2.');
        end
    elseif numel(varargin) >= 4
        gp = struct();
        gp.W1 = varargin{1};
        gp.b1 = varargin{2};
        gp.W2 = varargin{3};
        gp.b2 = varargin{4};
        if numel(varargin) >= 5 && ~isempty(varargin{5})
            gp.residualScale = varargin{5};
        end
    else
        error(['generate_channels: invalid arguments. ', ...
               'Expected (F, mask, A, W1, b1, W2, b2[, residualScale]) ', ...
               'or (F, mask, A, gen_params_struct).']);
    end

    W1 = gp.W1;    % [D_in x H]
    b1 = gp.b1;    % [1   x H]
    W2 = gp.W2;    % [H   x T]
    b2 = gp.b2;    % [1   x T]

    if isfield(gp, 'E_ch') && ~isempty(gp.E_ch)
        E_ch = gp.E_ch;              % [E_ch_dim x C]
    else
        E_ch = [];
    end

    if isfield(gp, 'E_mask') && ~isempty(gp.E_mask)
        E_mask = gp.E_mask;          % [E_mask_dim x C]
    else
        E_mask = [];
    end

    if isfield(gp, 'residualScale') && ~isempty(gp.residualScale)
        residualScale = gp.residualScale;
    else
        residualScale = single(0.2); % 新默认值（旧版 0.05）
    end

    if isfield(gp, 'useContext') && ~isempty(gp.useContext)
        useContext = logical(gp.useContext);
    else
        useContext = false;
    end

    % ---------- 2. 输入规格 ----------
    if ~isa(F_in, 'dlarray')
        F = dlarray(single(F_in));
    else
        F = F_in;
    end

    [B, T, C] = size(F);
    F_out = F;

    % ---------- 3. mask 统一为通道级 [1 x C] ----------
    if isa(abnormal_mask, 'dlarray')
        m_num = single(extractdata(abnormal_mask));
    else
        m_num = single(abnormal_mask);
    end

    if isvector(m_num)
        m_ch = reshape(m_num, 1, []);
    else
        m_ch = single(any(m_num == 1, 1));
    end

    if numel(m_ch) ~= C
        error('abnormal_mask must describe %d channels, but got %d elements.', ...
              C, numel(m_ch));
    end

    abnormal_channels = find(m_ch == 1);
    if isempty(abnormal_channels)
        return;
    end
    normal_mask = (m_ch == 0);

    % ---------- 4. 动态邻接矩阵：只从观测通道借信息 ----------
    if isa(A, 'dlarray')
        A_num = extractdata(A);
    else
        A_num = A;
    end
    A_num = single(A_num);

    A_dyn = A_num;
    A_dyn(:, ~normal_mask) = 0;       % 关闭对缺失通道的引用
    A_dyn(1:C+1:end) = 0;             % 去掉自环
    row_norm = sum(abs(A_dyn), 2) + 1e-8;
    A_dyn = A_dyn ./ row_norm;        % 按 |row| 归一，兼容 signed 邻接

    % ---------- 5. 观测通道池化上下文 (可选) ----------
    %   把"观测到的通道在时间轴上的平均特征"作为每个样本的全局线索，
    %   让 MLP 知道当前样本"剩下的干净信号大概长什么样"。
    ctx = [];
    if useContext
        obs_idx = find(normal_mask);
        if isempty(obs_idx)
            ctx = dlarray(zeros(B, T, 'single'));
        else
            ctx = mean(F(:, :, obs_idx), 3);     % [B x T x 1]
            ctx = reshape(ctx, [B, T]);          % [B x T]
        end
    end

    % ---------- 6. mask embedding (可选，与通道无关，批内共享) ----------
    e_m_row = [];
    if ~isempty(E_mask)
        if size(E_mask, 2) ~= C
            error('E_mask must be [E_mask_dim x C = %d], got [%d x %d].', ...
                  C, size(E_mask, 1), size(E_mask, 2));
        end
        m_vec = dlarray(single(m_ch(:)));        % [C x 1]
        e_m = E_mask * m_vec;                    % [E_mask_dim x 1]
        e_m_row = reshape(e_m, [1, numel(e_m)]); % [1 x E_mask_dim]
    end

    % ---------- 7. 逐缺失通道修复 ----------
    for ii = 1:numel(abnormal_channels)
        c = abnormal_channels(ii);

        % 7.1 signed A-Interp（dlarray 线性组合，不切断梯度）
        a_row = single(A_dyn(c, :));                 % [1 x C]
        a_row_dl = dlarray(reshape(a_row, [1 1 C])); % [1 x 1 x C]
        x_base = sum(F .* a_row_dl, 3);              % [B x T x 1]
        x_base = reshape(x_base, [B, T]);            % [B x T]

        % 7.2 组装条件输入 z_c ∈ [B x D_in]
        parts = {x_base};

        if useContext
            parts{end+1} = ctx;                      %#ok<AGROW>  [B x T]
        end

        if ~isempty(E_ch)
            if size(E_ch, 2) ~= C
                error('E_ch must be [E_ch_dim x C = %d], got [%d x %d].', ...
                      C, size(E_ch, 1), size(E_ch, 2));
            end
            e_c = reshape(E_ch(:, c), [1, size(E_ch, 1)]);  % [1 x E_ch_dim]
            e_c = repmat(e_c, B, 1);                        % [B x E_ch_dim]
            parts{end+1} = e_c;                             %#ok<AGROW>
        end

        if ~isempty(e_m_row)
            parts{end+1} = repmat(e_m_row, B, 1);           %#ok<AGROW>
        end

        z = cat(2, parts{:});                               % [B x D_in]

        % 7.3 维度检查（出错时给出明确提示）
        D_in_expect = size(W1, 1);
        if size(z, 2) ~= D_in_expect
            error(['generate_channels: MLP input dim mismatch. ', ...
                   'z is [%d x %d], W1 expects [%d x H]. ', ...
                   'Debug: T=%d, useContext=%d, E_ch_dim=%d, E_mask_dim=%d.'], ...
                   size(z, 1), size(z, 2), D_in_expect, T, ...
                   useContext, ...
                   size(E_ch, 1) * ~isempty(E_ch), ...
                   size(E_mask, 1) * ~isempty(E_mask));
        end

        if size(b1, 2) ~= size(W1, 2)
            error('b1 dim mismatch: b1 is [%d x %d], expected [1 x %d].', ...
                  size(b1, 1), size(b1, 2), size(W1, 2));
        end

        if size(W2, 1) ~= size(W1, 2)
            error('W2 dim mismatch: W2 is [%d x %d], first dim should be %d.', ...
                  size(W2, 1), size(W2, 2), size(W1, 2));
        end

        if size(W2, 2) ~= T
            error('W2 dim mismatch: W2 is [%d x %d], second dim should be T=%d.', ...
                  size(W2, 1), size(W2, 2), T);
        end

        % 7.4 两层 MLP
        h = z * W1 + b1;                  % [B x H]
        h = max(h, 0);                    % ReLU
        delta = h * W2 + b2;              % [B x T]

        % 7.5 残差叠加
        repaired = x_base + residualScale .* delta;

        % 7.6 写回 (dlarray 支持 subscripted assignment，梯度可传递)
        F_out(:, :, c) = repaired;
    end
end
