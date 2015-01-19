function yhat = myconstraintCB(param, model, x, y)
% slack resaling: argmax_y delta(yi, y) (1 + <psi(x,y), w> - <psi(x,yi), w>)
% margin rescaling: argmax_y delta(yi, y) + <psi(x,y), w>
%   if dot(y*x, model.w) > 1, yhat = y ; else yhat = - y ; end
%   if param.verbose
%     fprintf('yhat = violslack([%8.3f,%8.3f], [%8.3f,%8.3f], %3d) = %3d\n', ...
%             model.w, x, y, yhat) ;
%   end
    
    % y     = x
    % x     = current sample
    % model = current model
    % param = context parameter
    % yhat  = most violated constraint
    
    if all(model.w == 0)
        model.w = ones(length(model.w), 1);
        yhat = circshift(y, [1, -1]);
        
    else
        pos             = param.pos{x};
        neg             = param.neg{x};
        ScorePos        = cellfun(@(z) dot(model.w, z(:)), ...
            param.phi(x, pos));
        [Vpos, Ipos]    = sort(full(ScorePos'), 'descend');
        Ipos            = pos(Ipos);
        
        ScoreNeg        = cellfun(@(z) dot(model.w, z(:)), ...
            param.phi(x, neg));
        [Vneg, Ineg]    = sort(full(ScoreNeg'), 'descend');
        Ineg            = neg(Ineg);
        
        numPos          = length(pos);
        numNeg          = length(neg);
        n               = numPos + numNeg;
        
        NegsBefore = sum(bsxfun(@lt, Vpos, Vneg), 1);
        
        yhat        = nan * ones(n, 1);
        yhat((1:numPos) + NegsBefore) = Ipos;
%         yhat(end)   = Ipos;
        yhat(isnan(yhat)) = Ineg;
    end
    
end