%
% Created by Rui Zhao, on Sep 20, 2013. 
% This code is release under BSD license, 
% any problem please contact Rui Zhao rzhao@ee.cuhk.edu.hk
%
% Please cite as
% Rui Zhao, Wanli Ouyang, and Xiaogang Wang. 
% Learning Mid-level Filters for Person Re-identification. 
% In IEEE International Conference of Computer Vision and Pattern Recognition, 2014. 
%

test_trial = 1;
global dataset baseExp TRIAL gridstep patchsize par feat_dir salience_dir pwdist_dir ny nx dim

par = struct(...
    'dataset',                      'campus', ... %viper, campus
    'baseExp',                      'structsvm', ...
    'method',                       'test10', ... % 'mask_only', 'mask_sa1', 'mask_sa1_sa2', ...
    'TRIAL',                        test_trial, ... % test_trial
    'gridstep',                     4, ...
    'patchsize',                    10, ...
    'Nr',                           100, ...
    'sigma1',                       3.2, ... % VIPeR: 2.8 CAMPUS: 3.2
    'sigma2',                       0, ...
    'msk_thr',                      0.2, ...
    'norm_data',                    1, ...
    'new_feat',                     1, ...
    'new_match',                    1, ...
    'use_salience',                 1, ...
    'use_mask',                     1, ...
    'use_model',                    1, ...
    'ensemble',                     0, ...
    'alpha',                        [-1, 0, 0.4, 0.7, 0.6, 0],  ... %[-1, 0, 0.4, 0.7, 0.6, 0], ... %
    'L2',                           1, ...
    'swap',                         1 ...
    );

dataset     = par.dataset;
baseExp     = par.baseExp;
TRIAL       = par.TRIAL;
gridstep    = par.gridstep;
patchsize   = par.patchsize;
Nr          = par.Nr;

if par.L2 
    par.nor = 2;
else
    par.nor = 1;
end
nor  = par.nor;

switch par.method
    
    case 'mask_only'
        phiFun = @(x, y, m1, s1, s2, m2, par) (exp(-double(x).^2/par.sigma1^2).*m1).^nor;
        
    case 'test10'
        phiFun = @phiFun_new;
%         phiFun_update = @phiFun_new_update;
end
%
MS = 1; % multi-shot

project_dir = strcat(pwd, '\');
set_paths;
if par.norm_data
    normdata;
end
initialcontext;

%% extract dense features
if par.new_feat
    build_densefeature_general;
end

%% load features
% this may cost quite a while
feat_gal = zeros(dim, ny*nx, nPerson*2);
feat_prb = zeros(dim, ny*nx, nPerson*2);
hwait = waitbar(0, 'Loading data for testing ...');
for i = 1:nPerson
    load([feat_dir, 'feat', num2str(4*(i-1)+1), '.mat']);
    feat_gal(:, :, 2*i-1) = densefeat;
    load([feat_dir, 'feat', num2str(4*(i-1)+2), '.mat']);
    feat_gal(:, :, 2*i) = densefeat;
    load([feat_dir, 'feat', num2str(4*(i-1)+3), '.mat']);
    feat_prb(:, :, 2*i-1) = densefeat;
    load([feat_dir, 'feat', num2str(4*(i-1)+4), '.mat']);
    feat_prb(:, :, 2*i) = densefeat;
    waitbar(i/nPerson, hwait);
end
close(hwait)

%% compute or load patch matching distances
if par.new_match
    for i = 1:nPerson*2
        i
        for j = 1:nPerson*2
            [pwmap_gal{1, j}, pwmap_gal{2, j}, ~] = mutualmap(feat_gal(:, :, i), feat_prb(:, :, j));
            [pwmap_prb{1, j}, pwmap_prb{2, j}, ~] = mutualmap(feat_prb(:, :, i), feat_gal(:, :, j));
        end
        save([pwdist_dir, 'pwmap_gal_', num2str(i), '.mat'], 'pwmap_gal', '-v7.3');
        save([pwdist_dir, 'pwmap_prb_', num2str(i), '.mat'], 'pwmap_prb', '-v7.3');
    end
end

%% prepare training data
[phi, feat_prb, feat_gal, mask_prb, mask_gal, ...
    D_cell, P_cell, rdists_prb] = get_phi(ptrn, phiFun);

%% structural RankSVM training 
args = ' -c 1 -o 2 -v 1 ';
model = struct_ranksvm_train(phi, @myfeatureCB, @myconstraintCB, @mylossCB, args);

%% RankSVM model evaluation
pwdist_trn = -cellfun(@(x) dot(model.w, x(:)), phi)';
pwdist_notrn = -cellfun(@(x) dot(ones(1, numel(x)), x(:)), phi)';
[~, order_trn] = sort(pwdist_trn);

% show the number of negtive before positive (before and after training)
cmc_before = evaluate_pwdist(pwdist_notrn);
cmc_after = evaluate_pwdist(pwdist_trn);

plot([cmc_before, cmc_after]);
axis([0 100 0 1]);

%% prepare new augmented training data
% auc partitioning and hierachical clustering
K_stripe = 10; 
K_auc = 10;

if ~exist('mat/filter_all.mat', 'file')
    
    PAT_all = auc_partition(rdists_prb, mask_prb, K_stripe, K_auc);
    NOD_all = build_hierachical_tree(feat_prb, rdists_prb, PAT_all);
    
    % construct primitives for learning filters
    clear filters;
    for tk = 1:K_stripe
        for auc_level = 1:K_auc
            node = NOD_all{tk, auc_level};
            PAT = PAT_all{tk, auc_level};
            n = 0;
            for i = 1:numel(node)
                for j = 1:numel(node{i})
                    index_set = node{i}(j).set;
                    patset = PAT(index_set);
                    n = n + 1;
                    single_node.y = tk;
                    single_node.auclevel = auc_level;
                    single_node.info = [[patset.id];[patset.im]];
                    filters{tk, auc_level}(n) = single_node;
                end
            end
            display(['tk=', num2str(tk), ', ', 'auc=', num2str(auc_level)]);
        end
    end
    
    % deep node pruning
    clear filters_pruned;
    for tk = 1:K_stripe
        for auc_level = 1:K_auc
            n = 0;
            for i = 1:numel(filters{tk, auc_level})
                npat = size(filters{tk, auc_level}(i).info, 2);
                if npat < 300
                    n = n + 1;
                    filters_pruned{tk, auc_level}(n) = filters{tk, auc_level}(i);
                end
            end
        end
    end
    
    % select primitives among all filters
    filters_new = filters_pruned;
    
    %% disp('======= building weak rankers: svm training for each nodes ========');
    
    max_npat = 2500; % maximum number of patches in a cluster
    
    for tk = 1:K_stripe
        for auc_level = 1:K_auc
            wkRanker = filters_pruned{tk, auc_level};
            nfilter = numel(wkRanker);
            clear X_all y_all;
            fprintf('============================================\n')
            for i = 1:nfilter
                tic;
                clear positive positive_aux negative negative_aux;
                patset = wkRanker(i).info;
                npat = size(wkRanker(i).info, 2);
                positive = zeros(dim, npat);
                for j = 1:npat
                    positive(:, j) = feat_prb(:, patset(1, j), patset(2, j));
                end
                
                % samples from other nodes in the same stripe
                k_neg = 6;
                negative = zeros(dim, k_neg*npat);
                filters_other = cell2mat(filters_pruned(tk, setdiff(1:K_auc, auc_level)));
                info_other = filters_other.info;
                n_neg =  min(size(info_other, 2), k_neg*npat);
                neg_ind = randperm(size(info_other, 2), n_neg);
                for j = 1:n_neg
                    negative(:, j) = feat_prb(:, info_other(1, neg_ind(j)), info_other(2, neg_ind(j)));
                end
                
                % auxiliary positive set
                positive_aux = zeros(dim, npat);
                for j = 1:npat
                    [py, px] = ind2sub([ny, nx], patset(1, j));
                    pat_id_gal = sub2ind([ny, nx], py, P_cell{patset(2, j), patset(2, j)}(py, px));
                    positive_aux(:, j) = feat_gal(:, pat_id_gal, patset(2, j));
                end
                
                % auxiliary negative set
                K_neg = 20;
                sigma_s = 10;
                negative_aux = zeros(dim, K_neg*npat);
                n = 0;
                for j = 1:npat
                    [py, px] = ind2sub([ny, nx], patset(1, j));
                    orderset = order_trn(:, patset(2, j));
                    diffset = setdiff(orderset, patset(2, j));
                    sp = 1:length(diffset);
                    pdf = exp(-(sp.^2)./(sigma_s.^2));
                    ind = discretesample(pdf./sum(pdf), K_neg);
                    for k = (n+1):(n+K_neg)
                        pat_id_gal = ny * (P_cell{patset(2, j), diffset(ind(k-n))}(py, px) - 1) + py;
                        negative_aux(:, k) = feat_gal(:, pat_id_gal, diffset(ind(k-n)));
                    end
                    n = n + K_neg;
                end
                
                X_pos = cat(2, positive, positive_aux)';
                X_neg = cat(2, negative, negative_aux)';
                
                y_pos = ones(size(X_pos, 1), 1);
                y_neg = -ones(size(X_neg, 1), 1);
                
                X_all = cat(1, X_pos, X_neg);
                y_all = cat(1, y_pos, y_neg);
                
                if size(X_all, 1) < max_npat
                    model_f = svmtrain(y_all, X_all, '-t 0 -c 30 -b 1 -q');
                    filters_new{tk, auc_level}(i).w = model_f.SVs'*model_f.sv_coef;
                    filters_new{tk, auc_level}(i).b = -model_f.rho;
                else
                    filters_new{tk, auc_level}(i).w = zeros(dim, 1);
                    filters_new{tk, auc_level}(i).b = 0;
                end
                
                ti = toc;
                fprintf('Stripe %d/%d and auc level %d/%d: training %d/%d-th filter (%2.3f sec)\n', ...
                    tk, K_stripe, auc_level, K_auc, i, nfilter, ti);
            end
        end
    end
    
    % save filters
    save mat/filter_all.mat filters_new;
    
else
    load mat/filter_all.mat;
end

filters_select = filters_new(:, 6:end);
filters = cell2mat(filters_select(:)');
ytiks = round(linspace(0, ny, K_stripe+1));
for i = 1:numel(filters)
    filters(i).ys = (ytiks(filters(i).y)+1):ytiks(filters(i).y+1);
end

[phi_fire, norm_factor_prb, norm_factor_gal] ...
            = get_phi_fire(feat_prb, feat_gal, mask_prb, mask_gal, filters);

% training RankSVM model for re-ranking with original feature
rmodel_or = struct_ranksvm_train_rerank( ...
    phi, order_trn, @lossRS, @constrainRS, @featureRS, args);

% training RankSVM model for re-ranking with filtering resposne
rmodel_rr = struct_ranksvm_train_rerank( ...
    phi_fire, order_trn, @lossRS, @constrainRS, @featureRS, args);

% compute matching distance
score_or = cellfun(@(x) dot(rmodel_or.w, x(:)), phi);
score_rr = cellfun(@(x) dot(rmodel_rr.w, x(:)), phi_fire);

%% evaluation of RankSVM model with filtering response
w_or = 1.2;
w_rr = 1e5;%40; % weight for re-ranking weak ranking fires
cmc_rr = evaluate_pwdist(pwdist_trn - w_or.*score_or' - w_rr.*score_rr');

cmc_base = 1:length(cmc_before);
plot(cmc_base, cmc_before, 'b', cmc_base, cmc_after, 'g', cmc_base, cmc_rr, 'r')
%plot([cmc_before, cmc_after, cmc_rr]);
title('Training statistics');
axis([0 100 0 1]);

 %% testing
clear pwmap_prb_cell pwmap_gal_cell pwmap_all;
clear phi param D_cell P_cell mp_cell mg_cell sp_cell sg_cell;
clear phi_fire fires_prb fires_gal rfires_prb rfires_gal;
clear sfires_prb sfires_gal fires_prb_cell fires_gal_cell rmat_prb rmat_gal;

[phi_tst, feat_prb_tst, feat_gal_tst, mask_prb_tst, mask_gal_tst, ...
    D_cell_tst, P_cell_tst, rdists_prb_tst] = get_phi(ptst, phiFun, MS);

pwdist_tst = -cellfun(@(x) dot(model.w, x(:)), phi_tst)';
pwdist_notrn = -cellfun(@(x) dot(ones(1, numel(x)), x(:)), phi_tst)';

Sp = [4*(ptst-1)+3, 4*(ptst-1)+4]'; Sp = Sp(:);
Sg = [4*(ptst-1)+1, 4*(ptst-1)+2]'; Sg = Sg(:);

gsize = length(Sg);

if par.ensemble
    
    pwdist_ssvm = pwdist_tst;
    pwdist_dfeat = pwdist_tst;
    
    if strcmp(par.dataset, 'viper')
        load([pwdist_dir, 'MSCRmatch_VIPeR_f1_Exp007.mat']);
        load([pwdist_dir, 'txpatchmatch_VIPeR_f1_Exp007.mat']);
        load([pwdist_dir, 'wHSVmatch_VIPeR_f1_Exp007.mat']);
    elseif strcmp(par.dataset, 'campus')
        load([pwdist_dir, 'MSCRmatch_campus_f1_Exp007.mat']);
        load([pwdist_dir, 'txpatchmatch_campus_f1_Exp007.mat']);
        load([pwdist_dir, 'wHSVmatch_campus_f1_Exp007.mat']);
    else
        error(0);
    end
    
    pwdist_y = final_dist_y(Sg, Sp);
    pwdist_color = final_dist_color(Sg, Sp);
    pwdist_y = pwdist_y./repmat(max(pwdist_y, [], 1), gsize, 1);
    pwdist_color = pwdist_color./repmat(max(pwdist_color, [], 1), gsize, 1);
    pwdist_hist = final_dist_hist(Sg, Sp);
    pwdist_epitext = dist_epitext(Sg, Sp);
    
    pwdist = pwdist_ssvm + par.alpha(2).*pwdist_dfeat + ...
        par.alpha(3).*pwdist_y + par.alpha(4).*pwdist_color + ...
        par.alpha(5).*pwdist_hist + par.alpha(6).*pwdist_epitext;

else
    pwdist = pwdist_tst;
    
end

%% get the initial matching order

if MS
    pwdist_cell = mat2cell(pwdist, 2*ones(1, gsize/2), 2*ones(1, gsize/2));
    pwdist2 = cellfun(@(x) min(x(:)), pwdist_cell);
    CMC_MS = evaluate_pwdist(pwdist2); % curveCMC(CMC);
    [~, order_tst] = sort(pwdist2);
    
else
    CMC_SS = evaluate_pwdist(pwdist);
    [~, order_tst] = sort(pwdist);
end

[phi_fire_tst, ~, ~] ...
    = get_phi_fire(feat_prb_tst, feat_gal_tst, mask_prb_tst, mask_gal_tst, ...
                    filters, norm_factor_prb, norm_factor_gal);
%%
close all force;
w_or = 2.4;
w_rr = 2e5; %20;
base_rank = gsize/2;
new_order = order_tst;
for i = 1:size(pwdist_cell, 1)
    tmp_order = order_tst(1:base_rank, i);
    q_ind = [1, 2] + 2*(i -1);
    g_ind = [2*tmp_order-1, 2*tmp_order]';
    score_or = cellfun(@(x) dot(rmodel_or.w, x(:)), phi_tst(q_ind, g_ind(:)));
    score_rr = cellfun(@(x) dot(rmodel_rr.w, x(:)), phi_fire_tst(q_ind, g_ind(:)));
    score_or_cell = mat2cell(score_or, 2, 2*ones(1, base_rank));
    score_or_mean = cellfun(@(x) mean(x(:)), score_or_cell);
    score_rr_cell = mat2cell(score_rr, 2, 2*ones(1, base_rank));
    score_rr_mean = cellfun(@(x) mean(x(:)), score_rr_cell);

    dist = pwdist2(tmp_order, i)' - w_or.*score_or_mean - w_rr.*score_rr_mean;

    [~, ind] = sort(dist);
    new_order(1:base_rank, i) = tmp_order(ind);
end
match = (new_order == repmat(1:(gsize/2), [(gsize/2), 1]));
cmc_rr = cumsum(sum(match, 2)./(gsize/2));
plot(CMC_MS, '-bo'); hold on;
plot(cmc_rr, '-ro'); title('Testing statistics');
axis([0 30 0 1]);
display(['Rank1: ', num2str(CMC_MS(1)), ' AUC: ', num2str(sum(CMC_MS(1:base_rank)))]);
display(['Rank1: ', num2str(cmc_rr(1)), ' AUC: ', num2str(sum(cmc_rr(1:base_rank)))]);
