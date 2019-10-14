% BSOID_MASTER    A master script that includes all the functions to automatically classify rodent open field behaviors 
%                 upon DeepLabCut 3D-pose estimation.
%
%   Created by Alexander Hsu, Date: 072219
%   Contact ahsu2@andrew.cmu.edu

close all; clear all;
n = 6; % How many .csv files do you want to build your model on?
for i = 1:n
    %% Import data
    fprintf(sprintf('%s%s%s','Please select ',num2str(i),' DeepLabCut generated .csv file for training data. \n'));  [filecsv,pathcsv] = uigetfile; 
    filenamecsv = sprintf('%s%s',pathcsv,filecsv); data_struct = importdata(filenamecsv); rawdata{i} = data_struct.data;
    %% Low-pass filter based on user-defined likelihood threshold
    [MsTrainingData{i},perc_rect] = dlc_preprocess(rawdata{i},0.2); % Change 0.2 to your desired likelihood threshold
end
%% Run unsupervised learning algorithm (Gaussian Mixture Models) to identify different behaviors.
[f_10fps,tsne_feats,grp,llh,bsoid_fig] = bsoid_gmm(MsTrainingData,60,1); % Change 60 to your frame rate. Set 1 for 1 classifier for all .csv.
%% Build a Support Vector Machine (SVM) classifier based on your data
[OF_mdl,CV_amean,CV_asem,acc_fig] = bsoid_mdl(f_10fps,grp,0.2,100,200); % Change 40 and 200 to <= (training data size *0.2). 
% Save your model as a .mat file if it looks good.

%% Do this after you run the FFmpeg command
fprintf('Please select the folder containing FFmpeg generated frames from your 10fps video. \n');
PNGpath = uigetdir; PNGpath = sprintf('%s%s',PNGpath,'/');
fprintf('Please select output folder for GIF. \n');
GIFpath = uigetdir; GIFpath = sprintf('%s%s',GIFpath,'/');
% Assuming you trained on multiple sessions, select the session number corresponding to your video/frames
s_no = 3;
[t,B,b_ex] = action_gif2(PNGpath,grp(length(MsTrainingData{s_no})/(fps/10)*(s_no-1)-(s_no-1)+1:length(MsTrainingData{s_no})/(fps/10)*(s_no)-s_no),3,6,0.5,GIFpath);

%% Once you trained your action model
m = 1; % How many .csv do you want to test on?
for j = 1:m
    %% Import data
    fprintf(sprintf('%s%s%s','Please select ',num2str(j),' DeepLabCut generated .csv file for testing data. \n'));  [filecsv,pathcsv] = uigetfile; 
    filenamecsv = sprintf('%s%s',pathcsv,filecsv); data_test_struct = importdata(filenamecsv); rawdata_test{j} = data_test_struct.data;
    %% Low-pass filter based on user-defined likelihood threshold
    [MsTestingData{j},perc_rect] = dlc_preprocess(rawdata_test{j},0.2); % Change 0.2 to your desired likelihood threshold. Try matching training.
end
%% Classifier a test dataset that the algorithm has not seen before, no ground truth but can test against human observers
[labels,f_10fps_test] = bsoid_svm(MsTestingData,60,OF_mdl); % Change 60 to your frame rate. 
%% Compare with manually defined criteria
% [g_num,perc_unk,trans_p] = bsoid_mt(MsTestingData,60,24); % Change 24 to your set-up's pixel-to-centimeter. 
