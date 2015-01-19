function NOD_all = build_hierachical_tree(feature, rdists, PAT_all)

[ny, nx, tsize, ~] = size(rdists);

[K_stripe, K_auc] = size(PAT_all); %numel(PAT_auc);
%ytiks = round(linspace(0, ny, K_stripe+1));
% max_npat = 1000; 
for tk = 1:K_stripe
    %range_y = (ytiks(tk)+1):ytiks(tk+1);
%     clear PAT_stripe;
%     PAT_stripe = PAT_auc{tk};    
    for auc_level = 1:K_auc
    
%     if isempty(PAT_stripe)
%         continue;
%     else
%         PAT_stripe = PAT_stripe(randperm(numel(PAT_stripe), min(numel(PAT_stripe), 2.5*max_npat)));
%     end
    
%     PAT_cell = PAT_all{tk, auc_level};
%     nTree = ceil(numel(PAT_cell)/(max_npat+1));
%     tiks = ceil(linspace(0, numel(PAT_cell), nTree+1));
    
%     for tr = 1:nTree
        clear PAT;
%         ind_tree = (tiks(tr)+1):tiks(tr+1);
%         PAT = PAT_stripe(ind_tree);
        PAT = PAT_all{tk, auc_level};
        
        % compute affinity according to 1. spatial position, 2. auc statistics, 3. visual appearance
        disp('======= compute affinity matrix: x-deformation, color, sift ========');
        for i = 1:numel(PAT)
            PAT(i).color = feature(1:288, PAT(i).id, PAT(i).im);
            PAT(i).sift = feature(289:end, PAT(i).id, PAT(i).im);
            PAT(i).feat = feature(:, PAT(i).id, PAT(i).im);
        end
        
        % construct pairwise distance matrix
        A_color = sqDistance(cell2mat({PAT.color}));
        A_sift = sqDistance(cell2mat({PAT.sift}));
        [~, px] = cellfun(@(x) ind2sub([ny, nx], x), {PAT.id}, 'UniformOutput', false);
        A_disp = sqDistance(cell2mat(px));
        
        w_disp = 0.5./max(A_disp(:));
        w_color = 1./max(A_color(:));
        w_sift = 1./max(A_sift(:));
        
        A_cmb = w_disp.*A_disp + w_color.*A_color + w_sift.*A_sift;
        
        % construct hierarchical clustering tree
        disp('======= construct hierarchical clustering tree: graph agglomerative clustering ========');
        % params for tree
        ttn = numel(PAT);
        max_depth = 10;
        tree_order = 4;
        
        clear node;
        
        % hierarchical clustering for color & sift cue
        node{1}.set = 1:ttn; % 1-th node at 1-th level
        node{1}.parent = 0;
        n_parent = 1; % initial node
        i = 2; % initial level
        while n_parent > 0 && i <= max_depth
            chd = 0;
            for p = 1:n_parent
                index = node{i-1}(p).set;
                ids = cell2mat({PAT(index).im});
                % if the node contains few samples or few identities,discard it
                if numel(index) < 60 || sum(histc(ids, 1:tsize)~=0) < 3
                    continue;
                end
                K = min(ceil(numel(index)/tree_order*0.5), 30); % nearest neighbor
                label_cmb = gdlCluster(double(A_cmb(index, index)), tree_order, K, 1, false); % dglu
                group_num = max(label_cmb(:));
                if group_num > 1
                    for c = 1:group_num
                        chd = chd + 1;
                        node{i}(chd).set = index((label_cmb == c));
                        node{i}(chd).parent = p;
                    end
                end
            end
            n_parent = chd;
            i = i + 1;
        end
        
        NOD_all{tk, auc_level} = node;
        disp(['(', num2str(tk), ',', num2str(auc_level), ')']);
    end
end