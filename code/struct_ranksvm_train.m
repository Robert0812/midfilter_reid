function model = struct_ranksvm_train(phi, hFeature, hConstraint, hLoss, args)

% input: phi, hLoss, hFeature, hConstraint
% output: model

tsize = size(phi, 1);
param.phi = phi;
trnIds = 1:tsize;
patterns = num2cell(trnIds);
param.patterns = patterns;
param.pos = cellfun(@(x) find(trnIds == x), patterns, 'UniformOutput', false);
param.neg = cellfun(@(x) setdiff(trnIds, x), param.pos, 'UniformOutput', false);
param.labels = cellfun(@(x, y) cat(2, x, y), param.pos, param.neg, 'UniformOutput', false);

param.lossFn = hLoss;
param.constraintFn  = hConstraint;
param.featureFn = hFeature;
param.dimension = numel(param.phi{1});
param.verbose = 1;

model = svm_struct_learn(args, param);