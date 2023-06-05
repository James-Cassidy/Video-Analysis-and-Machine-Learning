clear all
close all

addpath testing_methods\

%% Import classification model
% Run the chosen classifier to save a model and import it here
classification_model =  load("HOG_SVM_Classifier.mat");
classification_model = classification_model.HOG_SVM_Classifier

%% Load test images

%Open testing image and convert to gray scale

TestingImages = imageDatastore('pedestrian/pedestrian','IncludeSubfolders',1,'LabelSource','foldernames');
NumAllTestingImages = numel(TestingImages.Files); % number of images

%% Load ground truth from file
ground_truth_dataset = Read_Test_Dataset("test.dataset");

predicted_pedestrians=[]; % holds all detection values
image_precision = []; % holds all precision values

%% HOG preprocessing
image = readimage(TestingImages, 1);
image=rgb2gray(image);
image = imresize(image, [160, 96]);
[hogl, visl] = extractHOGFeatures(image,'CellSize',[8,8]);
hogFeatureSize = length(hogl);
cellSize = [8,8];
testingFeatures = zeros(NumAllTestingImages, hogFeatureSize,'single');

%% Create and Open video object
vidObj = VideoWriter('Pedestrian_Detection.avi');
vidObj.FrameRate = 1;
open(vidObj);
predictions = [];
%% Sliding Window
% Run sliding window over evey test image in a single scale
for i=1:NumAllTestingImages
    current_image = i
    image = readimage(TestingImages, i);
    image=rgb2gray(image);
    imshow(image);
    figure(1)

    % Setup parameters
    samplingX = 161;
    samplingY = 97;
    window_count = 0;

    % Create M by 4 array for each image's ground truth values and remove NaN values
    ground_truth = ground_truth_dataset(i,:);
    ground_truth = rmmissing(ground_truth,2);
    ground_truth = reshape(ground_truth,[],4);
    
    for r=1:samplingX:size(image,1)
        for c= 1:samplingX:size(image,2)
            if (c+samplingY-1 <= size(image,2)) && (r+samplingX-1 <= size(image,1))

                window_count = window_count+1;
    
                %% Process image 
                %we crop the digit
                image3 = image(r:r+samplingX-1, c:c+samplingY-1);
                x_cord = size(image3,1);
                y_cord = size(image3,2);
    
                % we convert it into doubles from 0 to 1 and invert them
                image3 = 255 - image3;
                image3 = double(image3)/255;
    
                %All training examples were 160x90. We need to resample test images into a 160x90 image
                image3 = imresize(image3, [160, 96]);       
    
                %we reshape the digit into a vector
                image3 = preprocessDigit(image3);
%                  image = reshape(image,1,[]);
            testingFeatures(i,:) = extractHOGFeatures(image3,'CellSize', cellSize);
    
                %% Prediction and scores
                [prediction, scores] = predict(classification_model, testingFeatures(i,:));
                predictions = [predictions; prediction];    
                %% if there is a postive detection
                 if(prediction == 'pos')
                    detected = [x_cord, y_cord, c, r];
    
                    % precision of bounding boxes for each image
                    precision = bboxPrecisionRecall(detected, ground_truth, 0.5);
                    image_precision = [image_precision, precision];
    
                    % Save detections
                    predicted_pedestrians = [predicted_pedestrians, detected];

                    % draw bounding box of detection on image
                    figure(1)
                    hold on;
                    posision = [c,r,y_cord,x_cord];
                    rectangle('Position',posision,'EdgeColor','r', 'LineWidth', 2)
                    hold off;
                 end
                 frame = getframe(figure(1));
                 writeVideo(vidObj, frame);
            end
        end
    end
end

% Close video object
close(vidObj);

% Every detection of the sliding window
detections = predicted_pedestrians;

%% Evaluation
% average precision of detected bounding boxes for each image which is at least 50% of the ground truth bounding box
average_precision = mean(image_precision, 'all')
