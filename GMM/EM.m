clear,clf,clc;

M = 3;
epsilon = 1e-8;
flagMax = 100;

data = importdata('iris.mat');
X = data(:,1:end-1);
y = data(:,end);

[n,d]= size(X);

m(1,:) = [5 3 1 0];
m(2,:) = [6 3 4 1];
m(3,:) = [7 3 4 2];
s(:,:,1) = eye(4);
s(:,:,2) = eye(4);
s(:,:,3) = eye(4);
m0 = zeros(M,d,flagMax);
% m0(1,:) = [0 0];
% m0(2,:) = [0 0];
% m0(3,:) = [0 0];
s0 = zeros(d,d,M,flagMax);
% s0(:,:,1) = [0 0;0 0];
% s0(:,:,2) = [0 0;0 0];
% s0(:,:,3) = [0 0;0 0];
p = zeros(n,M,flagMax);
%EM begin
flag = 0;
while true
    flag = flag+1;
    m0(:,:,flag) = m;
    s0(:,:,:,flag) = s;
    p(:,:,flag) = Mstep(X, m, s);
    [m, s] = Estep(X, p(:,:,flag));
%     plot_gaussian_ellipsoid(m(3,:),s(:,:,3),2.447,[],gca,1+(cluster1Color-1)/flag);
%     plot_gaussian_ellipsoid(m(2,:),s(:,:,2),2.447,[],gca,cluster2Color/flag);
%     plot_gaussian_ellipsoid(m(1,:),s(:,:,1),2.447,[],gca,cluster3Color/flag);
    fprintf('%d loops done\n',flag);
    if flag==flagMax
        fprintf('Exceed %d looops!\n',flagMax);
        break;
    end
    if (sum(sum(abs(m-m0(:,:,flag))))<epsilon) && (sum(sum(sum(abs(s-s0(:,:,:,flag)))))<epsilon)
        break;
    end
end

l = zeros(n,1);
fmeasure = zeros(flag,1);
for i=1:flag
    l(max(p(:,:,i),[],2)==p(:,1,i)) = 1;
    l(max(p(:,:,i),[],2)==p(:,2,i)) = 2;
    l(max(p(:,:,i),[],2)==p(:,3,i)) = 3;
    fmeasure(i) = Fmeasure(y,l);
end

plot(1:flag,fmeasure, 'color', [1 0 1]);
axis([1,flag+1,0.5,1])

fprintf('------------------------------------------\n');
fprintf('Algorithm finished\n\n');
fprintf('%d iterations in total,\n\n', flag);
fprintf('data have been divided into %d clusters:\n\n', M);
for i=1:M
    fprintf('No.%d cluster:\n', i);
    fprintf('¦Ì = \n');
    disp(m0(i,:,flag));
    fprintf('¦² = \n');
    disp(s0(:,:,i,flag));
end
fprintf('F-measure = %f\n', fmeasure(end));