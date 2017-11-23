function [ mu, sigma ] = Estep(X, p)
%ESTEP Summary of this function goes here
%   Detailed explanation goes here

d = size(X,2);
[n, m] = size(p);
mu = p'* X ./ repmat(sum(p,1)',1,d);

s = sum(p,1);
sigma = zeros(d,d,m);
for k=1:m
    for i=1:n
        diff = X(i,:) - mu(k,:);
        sigma(:,:,k) = sigma(:,:,k) + p(i,k)*diff'*diff;
    end
    sigma(:,:,k) = sigma(:,:,k) ./ s(k);
end

end
