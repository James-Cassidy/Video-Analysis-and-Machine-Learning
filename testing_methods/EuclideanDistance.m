function dEuc = EuclideanDistance(sample1, sample2)
%EuclideanDistance finds euclidean distance between test and training images

sum = 0;
for i=1:size(sample1,2)
 sum = (sum + ((sample1(i)-sample2(i)).^2));
end
dEuc = sqrt(sum);

end