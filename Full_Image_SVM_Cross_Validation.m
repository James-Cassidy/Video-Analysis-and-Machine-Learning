%% Full Image and Single Vector Machine Classifier (SVM) OneVsOne

%   This function will make a datastore of positive and negative images of pedestrains
%   SVM classier used will be trained on the full image vectors
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

%% Training SVM Classifier using Full Image Vectors
%   Kernels used can be gaussian, rbf or polynomial

t = templateSVM('KernelFunction','gaussian');
Full_Image_SVM_CV_Classifier = fitcecoc(trainingImages, trainingLabels,'KFold', 5, 'Learners', t);

%% Cross Validation
%   K Folds set to 5
%   Divided into 5 subsets, each subset used for testing
%   so 2400 images used for training
%   and 600 used for testing
%   changes each time for another 4 iterations

predictedLabels = kfoldPredict(Full_Image_SVM_CV_Classifier);

cvTrainError = kfoldLoss(Full_Image_SVM_CV_Classifier);

% Accuracy of 5 fold Cross Validation
cvTrainAccuracy = 1 - cvTrainError;

%% Cross Validation Error and Accuracy in percentage
CV_Full_Image_SVM_Accuracy = cvTrainAccuracy*100

CV_Full_Image_SVM_ErrorRate = cvTrainError*100

%% Confusion Matrix with Labels

cm = confusionchart(trainingLabels,predictedLabels)

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

save Full_Image_SVM_CV_Classifier Full_Image_SVM_CV_Classifier