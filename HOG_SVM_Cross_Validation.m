%% Histogram of Oriented Gradients (HOG) and Single Vector Machine Classifier (SVM) OneVsOne

%   This function will make a datastore of positive and negative images of pedestrains
%   It will extract the HOG features of all the images in the train set
%   SVM classier used will be trained on the HOG feature vectors
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

trainingFeatures = zeros(NumAllTrainingImages, hogFeatureSize,'single');

for i = 1:NumAllTrainingImages
    img = readimage(trainSet,i);
    img = im2gray(img);
    trainingFeatures(i,:) = extractHOGFeatures(img,'CellSize', cellSize);
end

trainingLabels = trainSet.Labels;

%% Training SVM Classifier using HOG Features
%   Kernels used can be Gaussian, rbf or polynomial

t = templateSVM('KernelFunction','polynomial','Standardize',1);
HOG_SVM_CV_Classifier = fitcecoc(trainingFeatures, trainingLabels,'KFold', 5, 'Learners', t);

%% Cross Validation
%   K Folds set to 5
%   Divided into 5 subsets, each subset used for testing
%   so 2400 images used for training
%   and 600 used for testing
%   changes each time for another 4 iterations

predictedLabels = kfoldPredict(HOG_SVM_CV_Classifier);

cvTrainError = kfoldLoss(HOG_SVM_CV_Classifier);

% Accuracy of 5 fold Cross Validation
cvTrainAccuracy = 1 - cvTrainError;

%% Cross Validation Error and Accuracy in percentage
CV_HOG_SVM_Accuracy = cvTrainAccuracy*100

CV_HOG_SVM_ErrorRate = cvTrainError*100

%% Confusion Matrix with Labels
%   If presented with error, make sure "Image and it's HOG Features"
%   figure is closed   

cm = confusionchart(trainingLabels,predictedLabels);

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
%   Polynomial classifier saved as it is most accurate

save HOG_SVM_CV_Classifier HOG_SVM_CV_Classifier

toc