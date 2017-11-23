function [ f ] = Fmeasure( label, prediction )
%FMEASURE Summary of this function goes here
%   Detailed explanation goes here

a = 0;
b = 0;
c = 0;
n = length(label);
for i=1:n
    for j=1:n
        if label(i)==label(j)
            if(prediction(i)==prediction(j))
                a = a + 1;
            else
                b = b + 1;
            end
        elseif (prediction(i)==prediction(j))
            c = c + 1;                
        end
    end
end

f = 2*a/(2*a+b+c);

end

