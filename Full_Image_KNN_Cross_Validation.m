%% Full Image and K-Nearest Neighbour Classifier with K nearest neighbours (KNN)

%   This function will make a datastore of positive and negative images of pedestrains
%   KNN classier used will be trained on the full image vectors for pedestrian detection
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

TrainingImages = imageDatastore('images/images','IncludeSubfolders',1,'LabelSource','foldernames');
NumAllTrainingImages = numel(TrainingImages.Files); % number of images

rng(10); %   used for reproducability
trainSet = shuffle(TrainingImages);


% Process each training image to the correct format
trainingImages = zeros(NumAllTrainingImages, 15360);

for i = 1:NumAllTrainingImages
    img = readimage(trainSet,i);
    img = im2gray(img);
    img = reshape(img, size(img, 1) * size(img, 2), size(img, 3));
    trainingImages(i,:,:) = img;
end

trainingImages = double(trainingImages) / 255;
trainingLabels = trainSet.Labels;

%% KNN Classifier with Nearest Neighbours used
%   For KNN we agreed to use 1, 3, 5 and 10 for nearest neighbour
K = 10; % Number of neighbours
Full_Image_KNN_CV_Classifier = fitcknn(trainingImages, trainingLabels, 'KFold',5, 'NumNeighbors',K, 'Standardize',1);

%% Cross Validation
%   K Folds set to 5
%   Divided into 5 subsets, each subset used for testing
%   so 2400 images used for training
%   and 600 used for testing
%   changes each time for another 4 iterations

predictedLabels = kfoldPredict(Full_Image_KNN_CV_Classifier);

cvTrainError = kfoldLoss(Full_Image_KNN_CV_Classifier);

cvTrainAccuracy = 1 - cvTrainError;

%% Cross Validation Error and Accuracy in percentage
CV_Full_Image_KNN_Accuracy = cvTrainAccuracy*100

CV_Full_Image_KNN_ErrorRate = cvTrainError*100

%% Confusion Matrix with Labels and Scores

cm = confusionchart(trainingLabels, predictedLabels)

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

toc

save Full_Image_KNN_CV_Classifier Full_Image_KNN_CV_Classifier