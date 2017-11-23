function [ X, y ] = DataGenerator( N, M )

if M==2
    p = 2;
    n = N/2;
    X = [repmat([0, 0], n, 1); repmat([2, 1], N-n, 1)];
    X = X + randn(N,2)./p;
    y = [zeros(n,1); ones(n,1)];
else
    %这个数据就是报告中提到的真实数据
    data = importdata('data.mat');
    X = data(:,1:8);
    y = data(:,9);
end

end

