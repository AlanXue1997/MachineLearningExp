function [ p ] = Mstep( X, mu, sigma )
%MSTEP Summary of this function goes here
%   Detailed explanation goes here
pi = 3.1415926535898;
[m, d] = size(mu);
n = size(X,1);

p = zeros(n,m);
for j=1:n
    sum = 0;
    for i=1:m
        diff = X(j,:) - mu(i,:);
        p(j,i) = 1/((2*pi)^(d/2)*det(sigma(:,:,i))^0.5)*exp(-0.5*diff*pinv(sigma(:,:,i))*diff');
        sum = sum + p(j,i);
    end
    for i=1:m
        p(j,i) = p(j,i)/sum;
    end
end

end

