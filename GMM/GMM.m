function [ data ] = GMM(n, p, mu, sigma )
%GMM to generating data
%   Detailed explanation goes here

[m, d] = size(mu);
p = p./sum(p);
y = rand(n,1);
y = sum(y>(p'*triu(ones(m,m),0)),2);
data = zeros(n,d);
for i=1:n
    %S = sigma(:,:,y(i)+1)  % Э�������
    %M = [1; 1];      % ��ֵ
    %N = 1000;        % ���ݵ���
    %L = chol(S,'lower')
    %data(i,:) = (randn(1,d)*chol(sigma(:,:,y(i)+1),'lower') + mu(y(i)+1,:));
    data(i,:) = mvnrnd(mu(y(i)+1,:), sigma(:,:,y(i)+1),1);
end

data = [data y];

end