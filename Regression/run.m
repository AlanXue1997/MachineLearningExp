clear;clf;clc;

%==============��������==============
N = 30;%��������
M = 9;%����
lmd = 0.000000;%�������ϵ����û��������ʱΪ0

%==============��ȡ����==============
[data,y] = DataGenerator(N);

%==============��ʾ����==============
subplot(3, 2, 1);
hold on;
title('sin(2pi*x)');
sample_x = 0:0.01:1;
plot(data, y, '.', 'color', [0.7 0.7 0.7]);
plot(sample_x,sin(sample_x*2*pi), 'color', [0.3 0.3 0.3]);
xlabel(sprintf('Erms = %f', (sum((sin(data*2*pi)-y).^2))^0.5));
hold off;

%==============��������==============
%�������ݵ���Ϊ��εģ�
X = zeros(N, M);
X(:,1) = ones(N,1);
for i = 2:M+1,
    X(:,i) = X(:,i-1).*data;
end

%==============��С����==============
W = Normal(X, y, lmd);
sample_y = polyval(W(end:-1:1)', sample_x);
subplot(3, 2, 2);
hold on;
title('Least Squares');
plot(data, y, '.', 'color', [0.7 0.7 0.7]);
plot(sample_x,sin(sample_x*2*pi), 'color', [0.3 0.3 0.3]);
plot(sample_x,sample_y, 'color', [0 1 0.5]);
xlabel(sprintf('Erms = %f', (sum((X*W-y).^2))^0.5));
hold off;

%==============�ݶ��½�==============
alpha = 0.3;
epsilon = 0.00000000001;
W = randn(M+1, 1);%�����ֵ
J = [];%��¼ÿ�ε����ľ�����
[J(end+1), grad] = costFunction(W, X, y, lmd);
W = W - alpha.*grad;
[J(end+1), grad] = costFunction(W, X, y, lmd);
W = W - alpha.*grad;
while (J(end)-J(end-1))^2>epsilon,
    [J(end+1), grad] = costFunction(W, X, y, lmd);
    W = W - alpha.*grad;
end

%��ͼ
subplot(3, 2, 4);
semilogx(1:size(J,2),J, 'color', [1 0 1]);
hold on;
title('E');
xlabel(sprintf('Iteration = %d', size(J,2)));
hold off;
subplot(3, 2, 3);
hold on;
title('Gradient Descent');
text(0.63, 0.9, sprintf('epsilon=%.2e', epsilon));
text(0.63, 0.75, sprintf('lambda=%.2e', alpha));
plot(data, y, '.', 'color', [0.7 0.7 0.7]);
sample_y = polyval(W(end:-1:1)', sample_x);
plot(sample_x,sin(sample_x*2*pi), 'color', [0.3 0.3 0.3]);
plot(sample_x,sample_y, 'color', [1 0 1]);
xlabel(sprintf('Erms = %f', (sum((X*W-y).^2))^0.5));
hold off;

%==============�����ݶ�==============
epsilon2 = epsilon;
lambda = 0.3;
W = randn(M+1, 1)/10;
J = [];
[J(end+1), gradk] = costFunction(W, X, y, lmd);
pk = -gradk;
W = W + lambda*pk;
[J(end+1), gradk1] = costFunction(W, X, y, lmd);
beta = sum(gradk1.^2)/sum(gradk.^2);
pk1 = beta*pk-gradk1;
W = W + lambda*pk1;
gradk = gradk1;
pk = pk1;
while (J(end)-J(end-1))^2>epsilon2,
    [J(end+1), gradk1] = costFunction(W, X, y, lmd);
    beta = sum(gradk1.^2)/sum(gradk.^2);
    pk1 = beta*pk-gradk1;
    W = W + lambda*pk1;
    gradk = gradk1;
    pk = pk1;
end

%��ͼ
subplot(3, 2, 6);
semilogx(1:size(J,2),J, 'color', [1 0.5 0]);
hold on;
title('E');
xlabel(sprintf('Iteration = %d', size(J,2)));
sample_y = polyval(W(end:-1:1)', sample_x);
hold off;
subplot(3, 2, 5);
hold on;
title('Conjugate Gradient Descent');
text(0.63, 0.9, sprintf('epsilon=%.2e', epsilon2));
text(0.63, 0.75, sprintf('lambda=%.2e', lambda));
plot(data, y, '.', 'color', [0.7 0.7 0.7]);
plot(sample_x,sin(sample_x*2*pi), 'color', [0.3 0.3 0.3]);
plot(sample_x,sample_y, 'color', [1 0.5 0]);
xlabel(sprintf('Erms = %f', (sum((X*W-y).^2))^0.5));
hold off;
