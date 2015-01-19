function delta = mylossCB(param, y, ybar)
%   delta = double(y ~= ybar) ;
%   if param.verbose
%     fprintf('delta = loss(%3d, %3d) = %f\n', y, ybar, delta) ;
%   end
    
    % actually label y is given as the index of query data
  
    if length(y) == 1 
        q = y;
    else
        q = y(1);
    end
    
    NumPOS = length(param.pos{q});
    NumNEG = length(param.neg{q});
    
    NegsBefore = find(ybar == q) - 1;
    delta = NegsBefore / (NumPOS * NumNEG);
    
end