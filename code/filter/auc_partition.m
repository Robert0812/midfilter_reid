function PAT = auc_partition(rdists, mask, K_stripe, K_auc)

[ny, nx, tsize, ~] = size(rdists);

partial_n = ceil(0.3*tsize);
ytiks = round(linspace(0, ny, K_stripe+1));

for tk = 1:K_stripe
    % select curves: curves, and aux index
    range_y = (ytiks(tk)+1):ytiks(tk+1);
    
    % vertically constrained stripe
    stripe = false(ny, nx);
    stripe(range_y, :) = true;
    ind_stripe = repmat(stripe(:), 1, tsize);
    % foreground
    ind_mask = cell2mat(cellfun(@(x) reshape(x, [], 1), mask(:)', 'UniformOutput', false));
    ind_pat = ind_stripe & ind_mask;
    
    % record the preserved patch index
    info_pat = zeros(2, sum(ind_pat(:)));
    [info_pat(1, :), info_pat(2, :)] = find(ind_pat);
    
    % and their pAUC scores
    all_dists_pat = reshape(rdists, [], tsize);
    dists_pat = all_dists_pat(ind_pat, :)';
    partial_aucs = sum(dists_pat(1:partial_n, :), 1);
    
    % kmeans to separate into "K_auc" levels from low to high
%     [level_ids, auc_ctrs] = litekmeans(double(partial_aucs), K_auc);
%     [~, sctrs_ids] = sort(auc_ctrs);
    auc_tiks = linspace(min(partial_aucs), max(partial_aucs), K_auc+1);
    
    for auc_level = 1:K_auc
%         ind_auc = (level_ids == sctrs_ids(auc_level));
        ind_auc = (partial_aucs >= auc_tiks(auc_level)) & (partial_aucs < auc_tiks(auc_level+1));

        if sum(ind_auc) == 0
            PAT{tk, auc_level} = [];
        else
            info_auc = info_pat(:, ind_auc);
            score_auc = partial_aucs(ind_auc);
            for i = 1:size(info_auc, 2)
                PAT{tk, auc_level}(i).im = info_auc(2, i);
                PAT{tk, auc_level}(i).id = info_auc(1, i);
                PAT{tk, auc_level}(i).pauc = score_auc(i); %curves_auc(:, i);
                PAT{tk, auc_level}(i).color = []; %feature(1:288, info_cmb(1, i), info_cmb(2, i));
                PAT{tk, auc_level}(i).sift = []; %feature(289:end, info_cmb(1, i), info_cmb(2, i));
                PAT{tk, auc_level}(i).feat = []; %feature(:, info_cmb(1, i), info_cmb(2, i));
            end
        end
    end
    disp(['Partial AUC partitioning:', num2str(tk), '-th stripe ...']);
end
    
    