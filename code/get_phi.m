function [phi, feat_prb, feat_gal, mask_prb, mask_gal, ...
    D_cell, P_cell, rdists_prb] = get_phi(pidx, phiFun, MS)
% input: ptrn pwdist_dir par ny nx salience_dir phiFun
% output: phi tsize

if nargin < 3
    MS = 0;
end

global feat_dir pwdist_dir salience_dir par ny nx dim

% load gallery feature and probe feature for Structural SVM training
% gallery 1, *2 probe *3, 4
if MS
    tsize = length(pidx)*2;
    % idx for matching distance
    IdxP = [2*(pidx-1) + 1, 2*(pidx-1) + 2]'; IdxP = IdxP(:);
    IdxG = [2*(pidx-1) + 1, 2*(pidx-1) + 2]'; IdxG = IdxG(:);
    % idx for feature and mask
    prb_idx = [4*(pidx-1)+3, 4*(pidx-1)+4]'; prb_idx = prb_idx(:);
    gal_idx = [4*(pidx-1)+1, 4*(pidx-1)+2]'; gal_idx = gal_idx(:);
else
    tsize = length(pidx);
    % pwmap
    IdxP = 2*(pidx-1) + 1;
    IdxG = 2*(pidx-1) + 2;
    % feat & mask
    prb_idx = 4*(pidx-1)+3;
    gal_idx = 4*(pidx-1)+2;
end

% load image features
% if ~exist('feat_prb', 'var')
feat_prb = zeros(dim, ny*nx, tsize);
feat_gal = zeros(dim, ny*nx, tsize);
hwait = waitbar(0, 'Loading image features  ...');
for i = 1:tsize
    load([feat_dir, 'feat', num2str(prb_idx(i)), '.mat']);
    feat_prb(:, :, i) = densefeat;
    load([feat_dir, 'feat', num2str(gal_idx(i)), '.mat']);
    feat_gal(:, :, i) = densefeat;
    waitbar(i/tsize, hwait);
end
close(hwait);
% end

% load matching distance
for i = 1:tsize
    load([pwdist_dir, 'pwmap_prb_', num2str(IdxP(i)), '.mat']);
    load([pwdist_dir, 'pwmap_gal_', num2str(IdxG(i)), '.mat']);
    pwmap_prb_cell(:, i, :) = pwmap_prb(:, IdxG);
    pwmap_gal_cell(:, i, :) = pwmap_gal(:, IdxP);
    i
end

D_cell = squeeze(pwmap_prb_cell(1, :, :));
P_cell = squeeze(pwmap_prb_cell(2, :, :));

if MS
    refsize = length(pidx);
    dists_gal = cell2mat(reshape(pwmap_gal_cell(1, :, 1:2:end), 1, 1, tsize, refsize));
    dists_prb = cell2mat(reshape(pwmap_prb_cell(1, :, 2:2:end), 1, 1, tsize, refsize));
else
    refsize = tsize;
    dists_gal = cell2mat(reshape(pwmap_gal_cell(1, :, :), 1, 1, tsize, tsize));
    dists_prb = cell2mat(reshape(pwmap_prb_cell(1, :, :), 1, 1, tsize, tsize));
end
 
if par.use_salience
    rdists_gal = sort(dists_gal, 4);
    rdists_prb = sort(dists_prb, 4);
    maxdist_gal = rdists_gal(:, :, :, floor(refsize/2));
    maxdist_prb = rdists_prb(:, :, :, floor(refsize/2));
    lwdist = min(maxdist_gal(:));
    updist = max(maxdist_gal(:));
    salience_gal = (maxdist_gal-lwdist)./(updist-lwdist);
    salience_prb = (maxdist_prb-lwdist)./(updist-lwdist);
    
    salience_gal_cell = squeeze(mat2cell(salience_gal, ny, nx, ones(1, tsize)));
    salience_prb_cell = squeeze(mat2cell(salience_prb, ny, nx, ones(1, tsize)));
    sg_cell = repmat(salience_gal_cell', tsize, 1);
    sp_cell = repmat(salience_prb_cell, 1, tsize);
    
else 
    sp_cell = repmat({[]}, tsize, tsize);
    sg_cell = repmat({[]}, tsize, tsize);
end
 
 if par.use_mask
     load([salience_dir, 'posemask.mat']);
     
     mask_prb = squeeze(mat2cell(mask(:, :, prb_idx) >= par.msk_thr, ny, nx, ones(1, tsize)));
     mp_cell = repmat(mask_prb, 1, tsize);
     
     mask_gal = squeeze(mat2cell(mask(:, :, gal_idx) >= par.msk_thr, ny, nx, ones(1, tsize)));
     mg_cell = repmat(mask_gal', tsize, 1);
 end
 
 phi = cell(tsize, tsize);
 parfor i = 1:numel(D_cell)
     phi{i} = phiFun(D_cell{i}, P_cell{i}, mp_cell{i}, sp_cell{i}, sg_cell{i}, mg_cell{i}, par);
 end