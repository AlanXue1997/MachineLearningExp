function [ W ] = Normal( X, y, lmd )
%������С���˷��Ľ�

lambda = lmd;
W = pinv(X'*X + lambda*eye(size(X,2)))*X'*y;

end

