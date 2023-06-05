function prediction = KNNTesting(testImage, modelNN, K)
%KNNTesting returns the prediction of the accuracy of the model for K neighbours

neighbour = modelNN.neighbours;
label = modelNN.labels;
distances=zeros(size(label,1),2);

% Record Euclidean distance betweent the test and training images
for i=1:size(neighbour,1)
    distances(i,1) = EuclideanDistance(testImage, neighbour(i,:));
    distances(i,2) = label(i);
end

% Sort the distances from lowest to highest 
distances = sortrows(distances,1);
arrayOfLabels=[1,K];

% Add the K lowest distances to the array
for i=1:K
    arrayOfLabels(1,i) = distances(i,2);
end

% Prediction is the mode of the labels of the K training images with the smallest distances from the test image
prediction = mode(arrayOfLabels);

end