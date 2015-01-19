function [phi_fire, norm_factor_prb, norm_factor_gal] ...
    = get_phi_fire(feat_prb, feat_gal, mask_prb, mask_gal, ...
                    filters, norm_factor_prb, norm_factor_gal)

tsize = size(feat_prb, 3);

% filtering on training data
fires_prb = zeros(numel(filters), tsize);
fires_gal = zeros(numel(filters), tsize);
for r = 1:numel(filters)
    fires_prb(r, :) = wk_conv(filters(r), feat_prb, mask_prb);
    fires_gal(r, :) = wk_conv(filters(r), feat_gal, mask_gal);
    display(['Pre-fire: weak ranker ', num2str(r)]);
end
fires_prb((fires_prb < 0)) = 0;
fires_gal((fires_gal < 0)) = 0;

display('L2-normalizing along samples ...');

if nargin <= 5
    norm_factor_prb = sqrt(sum(fires_prb.^2, 2));
    norm_factor_gal = sqrt(sum(fires_gal.^2, 2));
end
efires_prb = fires_prb./repmat(norm_factor_prb(:), 1, tsize);
efires_gal = fires_gal./repmat(norm_factor_gal(:), 1, tsize);

efires_prb(isnan(efires_prb)) = 0; efires_prb(isinf(efires_prb)) = 0;
efires_gal(isnan(efires_gal)) = 0; efires_gal(isinf(efires_gal)) = 0;

display('L2-normalizing along filters ...');
rfires_prb = efires_prb./repmat(sqrt(sum(efires_prb.^2, 1)), numel(filters), 1);
rfires_gal = efires_gal./repmat(sqrt(sum(efires_gal.^2, 1)), numel(filters), 1);

sfires_prb = rfires_prb;
sfires_gal = rfires_gal;

ps = 1;
fires_prb_cell = mat2cell(sfires_prb.^ps, numel(filters), ones(1, tsize));
fires_gal_cell = mat2cell(sfires_gal.^ps, numel(filters), ones(1, tsize));
rmat_prb = repmat(fires_prb_cell(:), 1, tsize);
rmat_gal = repmat(fires_gal_cell, tsize, 1);

sigma_fire = 0.001;
fireFun = @(x, y) x.*exp(-(x-y).^2/(sigma_fire^2)).*y;
phi_fire = cellfun(fireFun, rmat_prb, rmat_gal, 'UniformOutput', false);