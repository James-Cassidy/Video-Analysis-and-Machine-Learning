function [images, labels] = loadTestingImagesAndLabels()
%loadTestingImagesAndLabels 

%% Load testing images

pedestrianImageFolder = '.\pedestrian\pedestrian\';

% Get list and number of all testing images 
negImageFilePattern = fullfile(pedestrianImageFolder, '*.jpg'); 
negImagesFile = dir(negImageFilePattern);
numberOfTestingImages = size(negImagesFile);

%% Testing images

testingImages = zeros( numel( negImagesFile ), 15360);

for i = 1 : numberOfTestingImages

    % Get path of each image
    baseFileName = negImagesFile(i).name;
    fullFileName = fullfile(negImagesFile(i).folder, baseFileName);

    % Read image
    testImage = imread(fullFileName);

    % Convert to grayscale
    testImage = im2gray(testImage);

    % Resize the test image to be the same size as the training images
    testImage = imresize(testImage, [160, 96]);

    % Reshape the image array
    testImage = reshape(testImage, size(testImage, 1) * size(testImage, 2), size(testImage, 3));

    % Add each image to the negative images array
    testingImages(i,:,:) = testImage;

    % Add a label with the same index position as its image
    testingLabels(i,:) = 1;

end

%% Return images and labels arrays

% Convert to double and rescale to [0,1]
images = double(testingImages) / 255;

labels = testingLabels;

end