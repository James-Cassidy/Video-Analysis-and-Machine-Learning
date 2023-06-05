clear all;
close all;

% Load in video by its file name and 
videoObj = VideoReader("Pedestrian_Detection.avi");
videoFrames = read(videoObj);
implay(videoFrames);