function [ J, grad ] = costFunction( W, X, y, lmd)

lambda = lmd;
m = length(y);
h = X * W;
J = sum((h-y).^2)/2+lambda*sum(W.^2);
grad = (X'*(h-y)+2*lambda*W)./m;

end

