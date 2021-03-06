function [ mu, sigma ] = Estep(X, p)
%ESTEP Summary of this function goes here
%   Detailed explanation goes here

d = size(X,2);
[n, m] = size(p);
%mu = p'* X ./ repmat(sum(p,1)',1,d);

mu = zeros(m,d);
for k=1:m
    sum = 0;
    for i=1:n
        if max(p(i,:))==p(i,k)
            sum = sum + p(i,k);
            mu(k,:) = mu(k,:) + X(i,:);
        end
    end
    mu(k,:) = mu(k,:) ./ sum;
end

sigma = zeros(d,d,m);
for k=1:m
    sum = 0;
    for i=1:n
        if max(p(i,:))==p(i,k)
            sum = sum + p(i,k);
            diff = X(i,:) - mu(k,:);
            sigma(:,:,k) = sigma(:,:,k) + p(i,k)*diff'*diff; 
        end
    end
    sigma(:,:,k) = sigma(:,:,k) ./ sum;
end

end
