clear,clf,clc;

n = 10000;
d = 3;
d1 = 2;
mu = zeros(1,d);
sigma = [
    100 70  0;
    70  100   70;
    0   70  100
    ];
data = mvnrnd(zeros(1,d), sigma,n);
plot3(data(:,1),data(:,2),data(:,3),'.');
plot_gaussian_ellipsoid(mu,sigma,2);
axis('equal');

data = data - repmat(sum(data)/n,n,1);
s = data'*data;
[V,D] = eig(s);
D = sum(D);
a = sum(D);
b = 0;
p = zeros(1,d);
for i=1:d1
    k = find(max(D)==D);
    p(k) = 1;
    b = b + D(k);
    D(k) = 0;
end

W = V(:,logical(p));

data1 = data*W;
%data1 = data*V;
data1 = data1*W';

plot3(data1(:,1),data1(:,2),data1(:,3),'.');

fprintf('%f\n',b/a);