function [images,labels] = convertImages(imageSet)
%convertImages converts image set into arrays of images and labels
%   Input: image set
%   Output: array of images and corresponding array of labels
%   Each image from the image set is preprocessed by converting it into
%   grayscale and reshaping the image into a row vector and finally adding
%   it into an image array. The corresponding label is then converted from
%   a categorical data type into an integer and added into a label array.

%% Convert images
% Parts of the code are taken from https://gitlab2.eeecs.qub.ac.uk/40267110/vaml/blob/benjamin-branch/loadTrainingImagesAndLabels.m

images = zeros( numel( imageSet.Files ), 15360);
labels = zeros( numel( imageSet.Labels ), 1);
[~, ~, G] = unique(imageSet.Labels);
NumImages = numel(imageSet.Files);

for i = 1 : NumImages

   % Get path of each image
    fullFileName = imageSet.Files{i};

    % Read image
    imageArray = imread(fullFileName);

    % Convert to grayscale
    imageArray = im2gray(imageArray);

    % Reshape the image array
    imageArray = reshape(imageArray, size(imageArray, 1) * size(imageArray, 2), size(imageArray, 3));

    % Add each image to the images array
    images(i,:,:) = imageArray;

    % Add a label with the same index position as its image
    % if image is neg, then -1
    % if image is pos, then 1
    if G(i) == 1
        labels(i,:) = -1;
    elseif G(i) == 2
        labels(i,:) = 1;
    end
   
end

% Rescale to [0,1]
images = images / 255;

end