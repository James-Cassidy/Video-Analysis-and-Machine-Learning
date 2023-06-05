tic
clear all;
close all;
addpath learning_methods\
addpath testing_methods\

%% Training Stage

% Load in the training and testing images and labels
TrainingImages = imageDatastore('images/images','IncludeSubfolders',1,'LabelSource','foldernames');
NumAllTrainingImages = numel(TrainingImages.Files); % number of images

rng(10); %   used for reproducability
[trainSet,testSet] = splitEachLabel(TrainingImages, 0.8, "randomized");
NumTrainingImages = numel(trainSet.Files);
NumTestingImages = numel(testSet.Files);

% Process each training image to the correct format
trainingImages = zeros(NumTrainingImages, 15360);

for i = 1:NumTrainingImages
    img = readimage(trainSet,i);
    img = im2gray(img);
    img = reshape(img, size(img, 1) * size(img, 2), size(img, 3));
    trainingImages(i,:,:) = img;
end

trainingImages = double(trainingImages) / 255;
trainingLabels = trainSet.Labels;

% Supervised training function that takes the training images and labels and infers a model
K = 3; % Number of neighbours
Full_Image_KNN_Classifier = fitcknn(trainingImages, trainingLabels,'NumNeighbors', K, 'Standardize', 1);

%% Testing 

% Process each testing image to the correct format
testingImages = zeros(NumTestingImages, 15360);

for i = 1:NumTestingImages
    img = readimage(testSet,i);
    img = im2gray(img);
    img = reshape(img, size(img, 1) * size(img, 2), size(img, 3));
    testingImages(i,:,:) = img;
end

testingImages = double(testingImages) / 255;
testingLabels = testSet.Labels;

% For each testing image, we obtain a prediction based on our trained model
predictedLabels = predict(Full_Image_KNN_Classifier, testingImages);

% Compared the predicted classification from our machine learning algorithm against the real labelling of the testing images to get the accuracy of the trained model
KNNAccuracy = (sum(predictedLabels == testingLabels)/numel(testingLabels))*100

%% Confusion Matrix with Labels
%   If Confusion Matrix returns an error, run this section on its own
%   Can only be ran as a section once previous code has ran

cm = confusionchart(testingLabels,predictedLabels)

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

%%
toc

save Full_Image_KNN_Classifier Full_Image_KNN_Classifier