figure(1),clf
clear,clc
dataColor=[204 204 204]/256;
cluster1Color=[0 184 230]/256;
cluster2Color=[255 166 33]/256;
cluster3Color=[0 184 0]/256;

flagMax = 1000;
epsilon = 1e-10;
n = 1000;
M = 3;
d = 2;
p = [10;10;50];
mu = [0, 0; 10, 2; 4, -6];
sigma(:,:,1) = [1, 0.2;
                0.2, 3];
sigma(:,:,2) = [1, 0.15;
                0.15, 1];
sigma(:,:,3) = [5, 2.2;
                2.2, 2];

% plot_gaussian_ellipsoid(mu(3,:),sigma(:,:,3),2.447);
% plot_gaussian_ellipsoid(mu(2,:),sigma(:,:,2),2.447);
figure(1)
hold on
axis('equal')
 
data = GMM(n, p, mu, sigma);
%display(data);
scatter(data(:,1),data(:,2),10,dataColor);
m = mu;
m(1,:) = [0 0];
m(2,:) = [0 1];
m(3,:) = [2 2];
s(:,:,1) = [1 0;0 1];
s(:,:,2) = [1 0;0 1];
s(:,:,3) = [1 0;0 1];
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
    p(:,:,flag) = Mstep(data(:,1:2), m, s);
    [m, s] = Estep(data(:,1:2), p(:,:,flag));
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

y = data(:,end);
l = zeros(n,1);
fmeasure = zeros(flag,1);
for i=1:flag
    plot_gaussian_ellipsoid(m0(3,:,i),s0(:,:,3,i),2.447,[],gca,cluster1Color+0.9*(1-cluster1Color)*(flag-i)/flag);
    plot_gaussian_ellipsoid(m0(2,:,i),s0(:,:,2,i),2.447,[],gca,cluster2Color+0.9*(1-cluster2Color)*(flag-i)/flag);
    plot_gaussian_ellipsoid(m0(1,:,i),s0(:,:,1,i),2.447,[],gca,cluster3Color+0.9*(1-cluster3Color)*(flag-i)/flag);
    drawnow
    l(max(p(:,:,i),[],2)==p(:,1,i)) = 1;
    l(max(p(:,:,i),[],2)==p(:,2,i)) = 2;
    l(max(p(:,:,i),[],2)==p(:,3,i)) = 3;
    fmeasure(i) = Fmeasure(y,l);
end
hold off
figure(2)
plot(1:flag,fmeasure, 'color', [1 0 1]);
axis([1,flag+1,0.5,1])

fprintf('------------------------------------------\n');
fprintf('Algorithm finished\n\n');
fprintf('%d iterations in total,\n\n', flag);
fprintf('data have been divided into %d clusters:\n\n', M);
for i=1:M
    fprintf('No.%d cluster:\n', i);
    fprintf('�� = \n');
    disp(m0(i,:,flag));
    fprintf('�� = \n');
    disp(s0(:,:,i,flag));
end
fprintf('F-measure = %f\n', fmeasure(end));