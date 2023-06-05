%% Histogram of Oriented Gradients (HOG) and K-Nearest Neighbour Classifier with K nearest neighbours (KNN)

%   This function will make a datastore of positive and negative images of pedestrains
%   It will extract the HOG features of all the images in the datastore
%   KNN classier used will be trained on the feature vectors for pedestrian detection
%   This script will use Cross Validation to measure accuracy etc

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

Images = imageDatastore('images/images','IncludeSubfolders',1,'LabelSource','foldernames');
NumAllImages = numel(Images.Files); % number of images

rng(10); %   used for reproducability
trainSet = shuffle(Images);

%%  Read Image and Extract it's HOG features (Example)

%   0 to 998 then negative (no pedestrians/ is background)
%   999 to 3001 then postive (has pedestrians)
%   k is the image you wish to use
%   Use value between 0 and 3001 for k
%   Will be random as the train images have been split into
%   test and train

k = 1624;
i = readimage(trainSet, k);
i = im2gray(i);
[hogl, visl] = extractHOGFeatures(i,'CellSize',[8,8]);

%%  Cell Size and Feature Size

cellSize = [8,8];
hogFeatureSize = length(hogl);

%%  Extract from all Training Images for HOG Features

trainingFeatures = zeros(NumAllImages, hogFeatureSize,'single');

for i = 1:NumAllImages
    img = readimage(trainSet,i);
    img = im2gray(img);
    trainingFeatures(i,:) = extractHOGFeatures(img,'CellSize', cellSize);
end

trainingLabels = trainSet.Labels;

%% KNN Classifier with Nearest Neighbours used
%   For KNN we agreed to use 1, 3, 5 and 10 for nearest neighbour

HOG_KNN_CV_Classifier = fitcknn(trainingFeatures, trainingLabels, 'KFold',5, 'NumNeighbors',1, 'Standardize', 1);

%% Cross Validation
%   K Folds set to 5
%   Divided into 5 subsets, each subset used for testing
%   so 2400 images used for training
%   and 600 used for testing
%   changes each time for another 4 iterations

predictedLabels = kfoldPredict(HOG_KNN_CV_Classifier);

cvTrainError = kfoldLoss(HOG_KNN_CV_Classifier);

cvTrainAccuracy = 1 - cvTrainError;

%% Cross Validation Error and Accuracy in percentage
CV_HOG_KNN_Accuracy = cvTrainAccuracy*100

CV_HOG_KNN_ErrorRate = cvTrainError*100

%% Confusion Matrix with Labels and Scores
%   If presented with error, make sure "Image and it's HOG Features"
%   figure is closed   

cm = confusionchart(trainingLabels, predictedLabels);

%% Confusion Matrix with Scores

Confusion_Matrix = confusionmat(trainingLabels,predictedLabels);

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
% K =1 agreed to save for classifier

save HOG_KNN_CV_Classifier HOG_KNN_CV_Classifier

toc