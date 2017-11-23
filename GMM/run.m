n = 1000;
M = 3;
d = 2;
p = [10;10;50];
mu = [0, 0; 10, 2; 4, -6];
sigma(:,:,1) = [1 0; 0 1];
sigma(:,:,2) = [1 0.15; 0.15 1];
sigma(:,:,3) = [5 2.2; 2.2 2];


% t = linspace(0,2*pi,1000);
% 
% [V,D] = eig(sigma(:,:,3))
% 
% theta0 = atan(V(2,1)/V(2,2));
% a=D(1,1);
% b=D(2,2);
% x = a*sin(t+theta0) + mu(3,1);
% y = b*cos(t) + mu(3,2);
% plot(x,y)
clf
plot_gaussian_ellipsoid(mu(3,:),sigma(:,:,3),2.447);
plot_gaussian_ellipsoid(mu(2,:),sigma(:,:,2),2.447);
hold on
axis('equal')
 
data = GMM(n, p, mu, sigma);
%display(data);
scatter(data(:,1),data(:,2));
hold off
m = mu;
s = sigma;
m0 = zeros(M, d);
s0 = zeros(d,d,M);
%EM begin
flag = 1;
while ~isequal(m,m0) || ~isequal(s,s0)
    flag = flag-1;
    m0 = m;
    s0 = s;
    p = Mstep(data(:,1:2), m, s);
    [m, s] = Estep(data(:,1:2), p);
    display(m);
    display(s);
    if flag==0
        display('Exceed 1 looops!');
        break;
    end
end
%display(cov(data(:,1:2)));