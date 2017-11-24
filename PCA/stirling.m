clear,clc;

calc = false;
n = 120;
d = [116 87];
precision = 0.9999;

data = zeros(n,d(1)*d(2));
for i=1:n
    p = imread(sprintf('data\\processed\\face_%d.gif',i));
    data(i,:) = p(:);
end

mu = repmat(sum(data)/n,n,1);

data = data - mu;

if calc
    s = data'*data;
    [V,D] = eig(s);
    D = sum(D);
    save VandD V D
else
    load VandD
end

a = sum(D);
b = 0;
p = zeros(1,d(1)*d(2));
d1 = 0;
while b < a*precision
%while d1 < 200
    d1 = d1 + 1;
    k = find(max(D)==D);
    p(k) = 1;
    b = b + D(k);
    D(k) = -1;
end
c = sum(D(~logical(p)));

W = V(:,logical(p));

data1 = data*W;
data2 = data1*W' + mu;

p = zeros(d(1),d(2));
for i=1:n
    p(:) = data2(i,:);
    imwrite(p,sprintf('data\\afterPCA\\face_%d.gif',i));
end
fprintf('d1 = %d, a = %f, b = %f, c = %f, b/c = %f\n',[d1, a, b, c, b/c]);