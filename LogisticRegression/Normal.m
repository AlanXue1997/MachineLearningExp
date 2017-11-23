function [ W ] = Normal( X, y, lmd )
%������С���˷��Ľ�

lambda = lmd;
m = size(X,2);
A = [zeros(1,m); [zeros(m-1, 1) eye(m-1)]];
W = pinv(X'*X + lambda*A)*X'*y;

end

