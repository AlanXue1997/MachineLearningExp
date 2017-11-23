function [ W ] = Normal( X, y, lmd )
%计算最小二乘法的解

lambda = lmd;
W = pinv(X'*X + lambda*eye(size(X,2)))*X'*y;

end

