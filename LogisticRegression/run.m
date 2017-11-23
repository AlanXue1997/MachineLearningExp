clear;clf;clc;

%==============调整参数==============
n = 1000;%样本数量
M = 2;%阶数
lmd = 0.001;%正则项的系数，没有正则项时为0

%==============获取数据==============
[data,Y] = DataGenerator(n, M);
N = floor(n*0.7);
train_set = data(1:N,:);%获得训练集
test_set = data((N+1):n, :);%获得测试集
y = Y(1:N);
test_y = Y(N+1:n);
%==============显示数据==============
subplot(2, 2, 1);
hold on;
% title('origin');
data0 = train_set(y==0, :);
data1 = train_set(y==1, :);
plot(data0(:,1), data0(:,2), '.', 'color', [0 0 1]);
plot(data1(:,1), data1(:,2), '.', 'color', [1 0 0]);
% %xlabel(sprintf('Erms = %f', (sum((sin(data*2*pi)-y).^2))^0.5));
hold off;

X = [ones(N,1) train_set];
X2 = [ones(n-N,1) test_set];

%==============梯度下降==============
alpha = 0.03;
epsilon = 0.000000001;
W = randn(M+1, 1)./1000;%随机初值
J = [];%记录每次迭代的均方差
J1 = [];
J2 = [];
[J(end+1), grad] = costFunction(W, X, y, lmd);
J1(end+1) = sum((X*W>0)==y)/N;
J2(end+1) = sum((X2*W>0)==test_y)/(n-N);%costFunction(W, X2, test_y, lmd);
W = W - alpha.*grad;
[J(end+1), grad] = costFunction(W, X, y, lmd);
J1(end+1) = sum((X*W>0)==y)/N;
J2(end+1) = sum((X2*W>0)==test_y)/(n-N);
W = W - alpha.*grad;
while (J(end)-J(end-1))^2>epsilon
    [J(end+1), grad] = costFunction(W, X, y, lmd);
    J1(end+1) = sum((X*W>0)==y)/N;
    J2(end+1) = sum((X2*W>0)==test_y)/(n-N);
    W = W - alpha.*grad;
    if length(J) > 10000
        break;
    end
end
fprintf('梯度下降：\n\t迭代次数：%d\n\tLost Function：%f\n\t训练集错误率：%f\n\t测试集错误率：%f\n', [length(J), J(end), J1(end), J2(end)]);
%绘图
subplot(2, 2, 2);
hold on;
%semilogx(1:size(J,2),J, 'color', [1 0 1]);
semilogx(1:size(J1,2),J1, 'color', [1 0 1]);
semilogx(1:size(J2,2),J2, 'color', [0 1 0]);
title('Gradient Descent');
xlabel(sprintf('Iteration = %d', size(J,2)));
hold off;
if M==2
    subplot(2, 2, 1);
    hold on;
    p1 = ezplot(sprintf('%f+(%f.*x)+(%f.*y)=0', W'));
    set(p1, 'Color', [1 0 1]);
    hold off;
end
    
%==============共轭梯度==============
epsilon2 = epsilon;
lambda = 0.3;
W = randn(M+1, 1)/1000;
J = [];
J1 = [];
J2 = [];
[J(end+1), gradk] = costFunction(W, X, y, lmd);
J1(end+1) = sum((X*W>0)==y)/N;
J2(end+1) = sum((X2*W>0)==test_y)/(n-N);
pk = -gradk;
W = W + lambda*pk;
[J(end+1), gradk1] = costFunction(W, X, y, lmd);
J1(end+1) = sum((X*W>0)==y)/N;
J2(end+1) = sum((X2*W>0)==test_y)/(n-N);
beta = sum(gradk1.^2)/sum(gradk.^2);
pk1 = beta*pk-gradk1;
W = W + lambda*pk1;
gradk = gradk1;
pk = pk1;
while (J(end)-J(end-1))^2>epsilon2
    [J(end+1), gradk1] = costFunction(W, X, y, lmd);
    J1(end+1) = sum((X*W>0)==y)/N;
    J2(end+1) = sum((X2*W>0)==test_y)/(n-N);
    beta = sum(gradk1.^2)/sum(gradk.^2);
    pk1 = beta*pk-gradk1;
    W = W + lambda*pk1;
    gradk = gradk1;
    pk = pk1;
end
fprintf('共轭梯度法：\n\t迭代次数：%d\n\tLost Function：%f\n\t训练集错误率：%f\n\t测试集错误率：%f\n', [length(J), J(end), J1(end), J2(end)]);
%绘图
subplot(2, 2, 3);
hold on;
semilogx(1:size(J1,2),J1, 'color', [1 0.5 0]);
semilogx(1:size(J2,2),J2, 'color', [0 1 0]);
title('Conjugate Gradient Descent');
xlabel(sprintf('Iteration = %d', size(J,2)));
hold off;
if M==2
    subplot(2, 2, 1);
    hold on;
    p2 = ezplot(sprintf('%f+(%f.*x)+(%f.*y)=0', W'));
    set(p2, 'Color', [1 0.5 0]);
    hold off;
end
%==============牛顿法==============
epsilon3 = epsilon;
W = randn(M+1, 1)/1000;%随机初值
J = [];%记录每次迭代的均方差
J1 = [];
J2 = [];
[J(end+1), grad, H] = costFunction(W, X, y, lmd);
J1(end+1) = sum((X*W>0)==y)/N;
J2(end+1) = sum((X2*W>0)==test_y)/(n-N);
W = W + pinv(H)*grad;
[J(end+1), grad, H] = costFunction(W, X, y, lmd);
J1(end+1) = sum((X*W>0)==y)/N;
J2(end+1) = sum((X2*W>0)==test_y)/(n-N);
W = W + pinv(H)*grad;
while (J(end)-J(end-1))^2>epsilon3
    [J(end+1), grad, H] = costFunction(W, X, y, lmd);
    J1(end+1) = sum((X*W>0)==y)/N;
    J2(end+1) = sum((X2*W>0)==test_y)/(n-N);
    W = W + pinv(H)*grad;
end
fprintf('牛顿法：\n\t迭代次数：%d\n\tLost Function：%f\n\t训练集错误率：%f\n\t测试集错误率：%f\n', [length(J), J(end), J1(end), J2(end)]);
%绘图
subplot(2, 2, 4);
hold on;
%semilogx(1:size(J,2),J, 'color', [0 0.5 1]);
semilogx(1:size(J1,2),J1, 'color', [0 0.5 1]);
semilogx(1:size(J2,2),J2, 'color', [0 1 0]);
title('Newton');
xlabel(sprintf('Iteration = %d', size(J,2)));
hold off;
if M==2
    subplot(2, 2, 1);
    hold on;
    p3 = ezplot(sprintf('%f+(%f.*x)+(%f.*y)=0', W'));
    set(p3, 'Color', [0 0.5 1]);
    title('data');
    % xlabel(sprintf('Erms = %f', (sum((X*W-y).^2))^0.5));
    hold off;
end