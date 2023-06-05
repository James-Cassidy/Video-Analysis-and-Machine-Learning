%% Histogram of Oriented Gradients (HOG) and K-Nearest Neighbour Classifier with K nearest neighbours (KNN)

%   This function will make a datastore of positive and negative images of pedestrains
%   It will extract the HOG features of all the images in the datastore
%   KNN classier used will be trained on the feature vectors for pedestrian detection

%%  Load training images from folder and save in datastore

%   folder named neg with no pedestrains/is background (998 images)
%   Folder named pos with pedestrains (2003 images)
%   There will be two datastores used,
%   One for training and one for testing
%   Will be split randomly 80:20

tic
clc;
clear;
close all;

TrainingImages = imageDatastore('images/images','IncludeSubfolders',1,'LabelSource','foldernames');
NumAllTrainingImages = numel(TrainingImages.Files); % number of images

rng(10); %   used for reproducability
[trainSet,testSet] = splitEachLabel(TrainingImages,0.8, "randomized");
NumTrainingImages = numel(trainSet.Files);
NumTestingImages = numel(testSet.Files);

%%  Read Image and Extract it's HOG features (Example)

%   0 to 998 then negative (no pedestrians/ is background)
%   999 to 3001 then postive (has pedestrians)
%   k is the image you wish to use
%   Use value between 0 and 3001 for k

k = 1337;
i = readimage(trainSet, k);
i = im2gray(i);
[hogl, visl] = extractHOGFeatures(i,'CellSize',[8,8]);

%%  Cell Size and Feature Size

cellSize = [8,8];
hogFeatureSize = length(hogl);

%%  Extract from all Training Images for HOG Features

trainingFeatures = zeros(NumTrainingImages, hogFeatureSize,'single');

for i = 1:NumTrainingImages
    img = readimage(trainSet,i);
    img = im2gray(img);
    trainingFeatures(i,:) = extractHOGFeatures(img,'CellSize', cellSize);
end

trainingLabels = trainSet.Labels;

%% KNN Classifier with Nearest Neighbours used
%   For KNN we agreed to use 1, 3, 5 and 10 for nearest neighbour

HOG_KNN_Classifier = fitcknn(trainingFeatures, trainingLabels, 'Standardize', 1,'NumNeighbors',1);

%% Test features

testingFeatures = zeros(NumTestingImages, hogFeatureSize,'single');
for i = 1:NumTestingImages
    imgt = readimage(testSet,i);
    imgt = im2gray(imgt);
    testingFeatures(i,:) = extractHOGFeatures(imgt,'CellSize', cellSize);
end

testingLabels = testSet.Labels;

%% What the model predicts the labels will be for unseen data
predictedLabels = predict(HOG_KNN_Classifier, testingFeatures);

%% Display the accuracy of the HOG KNN Classifier

HOG_KNN_Accuracy = (sum(predictedLabels == testingLabels)/numel(testingLabels))*100

%% Confusion Matrix with Labels
%   If Confusion Matrix returns an error, run this section on its own
%   Can only be ran as a section once previous code has ran

cm = confusionchart(testingLabels,predictedLabels);

%% Confusion Matrix with Scores in Console

Confusion_Matrix = confusionmat(testingLabels,predictedLabels);

Transposed_Confusion_Matrix = Confusion_Matrix';

diagonal = diag(Transposed_Confusion_Matrix);

sum_of_rows = sum(Transposed_Confusion_Matrix, 2);

%Precision
precision = diagonal ./ sum_of_rows
overall_precison = mean(precision)

sum_of_columns = sum(Transposed_Confusion_Matrix, 1);

%Recall
recall = diagonal ./ sum_of_columns'
overall_recall = mean(recall)

%F1 Score
f1_score = 2* ((overall_precison*overall_recall)/(overall_precison+overall_recall))

%% End
%   Save classifier
save HOG_KNN_Classifier HOG_KNN_Classifier

toc