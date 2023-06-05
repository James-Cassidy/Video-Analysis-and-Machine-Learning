tic
clc;
clear;
close all;


%% Load Training and Testing dataset
% The loadDataset function will output the training and testing dataset
% with an 80:20 split
[trainImages, trainLabels, testImages, testLabels] = loadDataset();

%% Apply PCA to the training dataset
% If there is no number of dimensions provided the
% PrincipalComponentAnalysis function will determine the number of 
% principal components needed to represent 95% of the total variance.
[eigenVectors, eigenvalues, meanX, Xpca_train] = PrincipalComponentAnalysis(trainImages);

%% Training SVM Classifier using PCA reduced training images
%   Kernels used can be Gaussian, rbf or polynomial

t = templateSVM('KernelFunction','gaussian');
PCA_SVM_Classifier = fitcecoc(Xpca_train, trainLabels,'Learners', t);

%% Apply PCA to the testing dataset

Xpca_test = (testImages - meanX) * eigenVectors;

%% What the model predicts the labels will be for unseen data

predictedLabels = predict(PCA_SVM_Classifier, Xpca_test);

%% Accuracy

PCA_SVM_Accuracy = (sum(predictedLabels == testLabels)/numel(testLabels))*100

%% Confusion Matrix with Labels
%   If Confusion Matrix returns an error, run this section on its own
%   Can only be ran as a section once previous code has ran

confusionchart(testLabels,predictedLabels)

%% Confusion Matrix with Scores in Console

Confusion_Matrix = confusionmat(testLabels,predictedLabels);

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

save PCA_SVM_Classifier PCA_SVM_Classifier

toc