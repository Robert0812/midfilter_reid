function psi = featureRS(param, x, y)
%   psi = sparse(y*x/2) ;
%   if param.verbose
%     fprintf('w = psi([%8.3f,%8.3f], %3d) = [%8.3f, %8.3f]\n', ...
%             x, y, full(psi(1)), full(psi(2))) ;
%   end
% end

% q  = index of the query sample
% y  = the ordering to compute
% pos = indices of relevant results for q
% neg = indices of irrelevant results for q

% psi = the joint feature map for query sample q

% x     = index of query sample
% y     = the ordering to compute
% param = parameters
%       param.pos   = indices of relevant samples for x
%       param.neg   = indices of irrelevant samples for x
%       param.phi     = pairwise feature vector

    phi     = param.phi;
%     pos     = param.pos{x};
%     neg     = param.neg{x};
%     yp      = ismember(y, pos);
%     NumPOS  = sum(yp);
%     yn      = ~yp;
%     NumNEG  = sum(yn);
    pos = param.pos{x};
    NumPOS = length(pos);
    neg = param.neg{x};
    NumNEG = length(neg);
    N = NumPOS + NumNEG;
    
    if length(y) == 1
        % y is groundtruth
        label       = -ones(1, N);
        label(1)    = 1;
    else
        yp          = ismember(y, pos);
        yn          = ~yp;
        label = 2*bsxfun(@lt, find(yp(:)), find((yn(:))')) - 1;    
    end
    
    psi = zeros(size(phi{1}));
    for i = 1:NumPOS
        for j = 1:NumNEG
            psi = psi + label(i, j)*(phi{x, pos(i)} - phi{x, neg(j)});
        end
    end
    psi = sparse(double(psi(:))./(NumPOS*NumNEG));
end