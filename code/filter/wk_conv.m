function fires = wk_conv(wkRanker, feature, mask)
% use weak ranker to convolve a set of features with a specific map
% INPUT
%   wkRanker: wkRankers containing svm training result
%   feature:    a set of features (column wise)
%   mask:       a set of mask (a set of cells)
%
% OUTPUT
%   fires:  an array of fires for each sample
%

[~, ~, tsize] = size(feature);
[ny, nx] = size(mask{1});
stripe = false(ny, nx);

% wkRanker.ys = round(wkRanker.ys(1)/38*ny):round(wkRanker.ys(end)/38*ny);

stripe(wkRanker.ys, :) = true;
n_pat = sum(stripe(:));
cnts = histc(wkRanker.info(1, :), 1:nx*ny);
spatial_pdf = cnts(stripe)./sum(cnts);

svm_w = repmat(wkRanker.w, [1, n_pat, tsize]);

ind_mask = cell2mat(cellfun(@(x) reshape(x, [], 1), mask(:)', 'UniformOutput', false));
% tmp_fire = bsxfun(@times, spatial_pdf(:), squeeze(sum(svm_w.*feature(:, stripe, :), 1)));
% fires = sum(tmp_fire.*ind_mask(stripe, :), 1) + wkRanker.b;

tmp_fire = squeeze(sum(svm_w.*feature(:, stripe, :), 1));
fires = max(tmp_fire.*ind_mask(stripe, :), [], 1) + wkRanker.b;