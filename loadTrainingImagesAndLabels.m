function [images, labels] = loadTrainingImagesAndLabels()
%loadTrainingImagesAndLabels 

%% Load training images

negImageFolder = '.\images\images\neg\';
posImageFolder = '.\images\images\pos\';

% Get list and number of all negative images 
negImageFilePattern = fullfile(negImageFolder, '*.jpg'); 
negImagesFile = dir(negImageFilePattern);
numberOfNegImages = size(negImagesFile);

% Get list and number of all positive images 
posImageFilePattern = fullfile(posImageFolder, '*.jpg'); 
posImagesFile = dir(posImageFilePattern);
numberOfPosImages = size(posImagesFile);

%% Negative images

negImages = zeros(numel( negImagesFile ), 15360);

for i = 1 : numberOfNegImages

    % Get path of each image
    baseFileName = negImagesFile(i).name;
    fullFileName = fullfile(negImagesFile(i).folder, baseFileName);

    % Read image
    imageArray = imread(fullFileName);

    % Convert to grayscale
    imageArray = im2gray(imageArray);

    % Reshape the image array
    imageArray = reshape(imageArray, size(imageArray, 1) * size(imageArray, 2), size(imageArray, 3));

    % Add each image to the negative images array
    negImages(i,:,:) = imageArray;

    % Add a label with the same index position as its image
    negLabels(i,:) = 0;

end

%% Positive Images

posImages = zeros( numel( posImagesFile ), 15360);

for i = 1 : numberOfPosImages

   % Get path of each image
    baseFileName = posImagesFile(i).name;
    fullFileName = fullfile(posImagesFile(i).folder, baseFileName);

    % Read image
    imageArray = imread(fullFileName);

    % Convert to grayscale
    imageArray = im2gray(imageArray);

    % Reshape the image array to be same size as the testing images
    imageArray = reshape(imageArray, size(imageArray, 1) * size(imageArray, 2), size(imageArray, 3));

    % Add each image to the negative images array
    posImages(i,:,:) = imageArray;

    % Add a label with the same index position as its image
    posLabels(i,:) = 1;

end

%% Return images and labels arrays

% Concatenate the positive and negative image and label arrays
images = cat(1,negImages,posImages);
labels = cat(1,negLabels,posLabels);

% Convert to double and rescale to [0,1]
images = double(images) / 255;

end