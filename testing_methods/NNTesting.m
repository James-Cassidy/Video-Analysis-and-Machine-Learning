function prediction = NNTesting(testImage, modelNN)
%NNTesting returns the prediction of the accuracy of the model for the nearest neighbour

neighbour = modelNN.neighbours;
labels = modelNN.labels;

distances = zeros(size(labels,1),2);

% Record Euclidean distance betweent the test and training images
for i=1:size(neighbour,1)
    distances(i,1) = EuclideanDistance(testImage, neighbour(i,:));
    distances(i,2) = labels(i);
end

% Sort the distances from lowest to highest 
distances = sortrows(distances,1);

% Prediction is the label of the training image with the lowest distance from the test image
prediction = distances(1,2);
end