n = 120;

for i=1:n
    p = imread(sprintf('data\\stirling\\face (%d).gif',i));
    imwrite(p(1:3:346,1:3:260),sprintf('data\\processed\\face_%d.gif',i));
end