function [ J, grad, H ] = costFunction( W, X, y, lmd)

lambda = lmd;
m = length(y);
n = size(X,2);
h = ones(m,1)./(1+exp(X*W));
J = -(sum(y.*log(h) + (ones(m,1)-y).*log(ones(m,1)-h)))/m;
%grad = (X'*(h-y)+2*lambda*W)./m;
grad = -(X'*(h-y))./m + lambda*[0; W(2:end)];

H = zeros(n,n);
for i=(1:n)
    for j=(1:n)
        SUM = 0;
        for t=(1:m)
            SUM = SUM + h(t)*(h(t)-1)*X(t,i)*X(t,j);
        end
        H(i,j) = SUM./m;
    end
    if i~=1
        H(i,i) = H(i,i) + lambda;
    end
end

end

