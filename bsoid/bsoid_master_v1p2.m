% BSOID_MASTER_V1P2    A master script for B-SOiD v1.2 to run adaptive high-pass filter, segmentation/assignment, and build a model for 
%                      to classify future dataset behaviors based on pose.
%
%   Created by Alexander Hsu, Date: 021920
%   Contact ahsu2@andrew.cmu.edu

close all; clear all;
n = 2; % How many .csv files do you want to build your model on?
for i = 1:n
    %% Import data
    fprintf(sprintf('%s%s%s','Please select ',num2str(i),' DeepLabCut generated .csv file for training data. \n'));  
    cd /your/dlc/output/folder/ % make sure you change this to the folder you have saved the .csv files for pose estimation
    [filecsv,pathcsv] = uigetfile('*.csv'); % show you only .csv files for user-friendliness
    filenamecsv = sprintf('%s%s',pathcsv,filecsv); data_struct = importdata(filenamecsv); rawdata{i} = data_struct.data; % import data as matrix in a cell
    %% Adaptive high-pass filter based on data for parts that are easily occluded
    [MsTrainingData{i},perc_rect] = adp_filt(rawdata{i});
end

%% Segment the groups based on natural statistics of input features, refer to our paper for feature list
[feats,tsne_feats,grp,llh,bsoid_fig] = bsoid_assign(MsTrainingData,60,1); % Change 60 to your frame rate. Set 1 for a compiled space for all .csv.

%% Build a Support Vector Machine (SVM) classifier based on your data
[behv_mdl,cv_amean] = bsoid_mdl2(feats,grp,0.2); % Change 0.2 to desired ratio of held out data to test the classifier on. 
% Save your model as a .mat file if it looks good.

%% Do this after you run the FFmpeg command
fprintf('Please select the folder containing FFmpeg generated frames from your 10fps video. \n');
PNGpath = uigetdir; PNGpath = sprintf('%s%s',PNGpath,'/');
fprintf('Please select output folder for GIF. \n');
GIFpath = uigetdir; GIFpath = sprintf('%s%s',GIFpath,'/');
% Assuming you trained on multiple sessions/.csv, select the .csv number (order in which you selected up top) corresponding to your video/frames
s_no = 1; % change 1 to 2 if you extracted the video frames from the second .csv you imported, as an example 
[t,B,b_ex] = action_gif2(PNGpath,grp(length(MsTrainingData{s_no})/(60/10)*(s_no-1)-(s_no-1)+1:length(MsTrainingData{s_no})/(60/10)*(s_no)-s_no),5,3,0.5,GIFpath);

%% Once you trained your action model
m = 1; % How many .csv do you want to test on?
for j = 1:m
    %% Import data
    fprintf(sprintf('%s%s%s','Please select ',num2str(j),' DeepLabCut generated .csv file for testing data. \n'));  
    cd /your/dlc/output/folder/ % make sure you change this to the folder you have saved the .csv files for pose estimation
    [filecsv,pathcsv] = uigetfile('*.csv'); % show you only .csv files for user-friendliness
    filenamecsv = sprintf('%s%s',pathcsv,filecsv); data_test_struct = importdata(filenamecsv); rawdata_test{j} = data_test_struct.data;
    %% Adaptive high-pass filter based on data for parts that are easily occluded
    [MsTestingData{j},perc_rect] = adp_filt(rawdata_test{j});
end
%% Classifier a test dataset that the algorithm has not seen before, no ground truth but can test against human observers
%%% As long as the distance from view is similar, this behavioral model can predict action based on pose with a different frame rate than the training.
%%% For instance, I built a SVM classifier based on 60 fps and generalized the prediction to a 200fps video behaviors based on pose.
[labels,f_10fps_test] = bsoid_svm(MsTestingData,60,behv_mdl); % Change 60 to your frame rate.   

%% In addition, you can play with frame-shifted machine learning prediction for detection of behavioral start up to camera frame rate
[labels_fsALL,f_10fps_fs] = bsoid_fsml(MsTestingData,60,behv_mdl); % Change 60 to your frame rate.

