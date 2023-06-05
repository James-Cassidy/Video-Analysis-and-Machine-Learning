function [trainImages, trainLabels, testImages, testLabels] = loadDataset()
%loadDataset outputs training and testing images and labels
%   The training image data set is loaded and split into a training and
%   testing set. Each set is then preprocessed and converted into arrays of images and
%   labels.


%% Load training image data set
% Code from this section taken from https://gitlab2.eeecs.qub.ac.uk/40267110/vaml/blob/james/HOG_SVM_Training.m

TrainingImages = imageDatastore('images/images','IncludeSubfolders',1,'LabelSource','foldernames');

rng(10); %   used for reproducability
[trainSet,testSet] = splitEachLabel(TrainingImages,0.8, "randomized");

%% Convert training images into array

[trainImages,trainLabels] = convertImages(trainSet);


%% Convert testing images into array

[testImages,testLabels] = convertImages(testSet);


end