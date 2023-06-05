tic
clc;
clear;
close all;

%% Load dataset
TrainingImages = imageDatastore('images/images','IncludeSubfolders',1,'LabelSource','foldernames');
NumAllTrainingImages = numel(TrainingImages.Files); % number of images

rng('default'); %   used for reproducability
trainSet = shuffle(TrainingImages);

%% Process each image from dataset
trainingImages = zeros(NumAllTrainingImages, 15360);

for i = 1:NumAllTrainingImages
    img = readimage(trainSet,i);
    img = im2gray(img);
    img = reshape(img, size(img, 1) * size(img, 2), size(img, 3));
    trainingImages(i,:,:) = img;
end
trainingImages = double(trainingImages) / 255;
trainingLabels = trainSet.Labels;

%% Apply PCA to images
[eigenVectors, eigenvalues, meanX, trainingFeatures] = PrincipalComponentAnalysis(trainingImages);

%% KNN Classifier with Nearest Neighbours used
%   For KNN we agreed to use 1, 3, 5 and 10 for nearest neighbour

PCA_KNN_CV_Classifier = fitcknn(trainingFeatures, trainingLabels, 'KFold',5, 'NumNeighbors',1);

%% Cross Validation
%   K Folds set to 5
%   Divided into 5 subsets, each subset used for testing
%   so 2400 images used for training
%   and 600 used for testing
%   changes each time for another 4 iterations

predictedLabels = kfoldPredict(PCA_KNN_CV_Classifier);

cvTrainError = kfoldLoss(PCA_KNN_CV_Classifier);

cvTrainAccuracy = 1 - cvTrainError;

%% Cross Validation Error and Accuracy in percentage
CV_PCA_KNN_Accuracy = cvTrainAccuracy*100

CV_PCA_KNN_ErrorRate = cvTrainError*100

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

save PCA_KNN_CV_Classifier PCA_KNN_CV_Classifier

toc