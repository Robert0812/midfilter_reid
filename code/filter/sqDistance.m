function D = sqDistance(X, Y)
if nargin < 2
    Y = X;
end
D = bsxfun(@plus,dot(X,X,1)',dot(Y,Y,1))-2*(X'*Y);