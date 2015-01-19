function rmodel_rr = struct_ranksvm_train_rerank(...
    phi_fire, order_trn, lossfn, constrainfn, featurefn, args)

tsize = size(phi_fire, 1);
base_rank = 30;

n = 0; 
for i = 1:tsize
    initial_rank = find(order_trn(:, i) == i);
    if initial_rank < base_rank
        n = n + 1;
        query_index(n) = i;
        rparam.patterns{n} = n;
        rparam.pos{n} = i;
        rparam.neg{n} = setdiff(order_trn(1:base_rank, i)', i);
        rparam.labels{n} = [rparam.pos{n}, rparam.neg{n}];
    end
end
rparam.lossFn = lossfn;
rparam.constraintFn = constrainfn;
rparam.featureFn = featurefn;
rparam.verbose = 1;

rparam.phi = phi_fire(query_index, :);
rparam.dimension = numel(rparam.phi{1});
rmodel_rr = svm_struct_learn(args, rparam);
