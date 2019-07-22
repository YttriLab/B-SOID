% BSOID_MASTER    A master script that includes all the functions to automatically classify rodent open field behaviors 
%                 upon DeepLabCut 3D-pose estimation.
%
%   Created by Alexander Hsu, Date: 072219
%   Contact ahsu2@andrew.cmu.edu

close all; clear all;
n = 6; % How many .csv files do you want to build your model on?
for i = 1:n
    %% Import data
    fprintf('Please select DeepLabCut generated .csv file for training data. \n'); [filecsv,pathcsv] = uigetfile; 
    filenamecsv = sprintf('%s%s',pathcsv,filecsv); data_struct = importdata(filenamecsv); rawdata{i} = data_struct.data;
    %% Low-pass filter based on user-defined likelihood threshold
    [MsTrainingData{i},perc_rect] = dlc_preprocess(rawdata{i},0.2);
end
%% Run unsupervised learning algorithm (Gaussian Mixture Models) to identify different behaviors.
[f_10fps,tsne_feats,grp,llh,bsoid_fig] = bsoid_gmm(MsTrainingData,60,1);
%% Build a Support Vector Machine (SVM) classifier based on your data
[OF_mdl,CV_amean,CV_asem,acc_fig] = bsoid_mdl(f_10fps,grp,0.2,70,200);
m = 1; % How many .csv do you want to test on?
for j = 1:m
    %% Import data
    fprintf('Please select DeepLabCut generated .csv file for testing data. \n'); [filecsv,pathcsv] = uigetfile; 
    filenamecsv = sprintf('%s%s',pathcsv,filecsv); data_test_struct = importdata(filenamecsv); rawdata_test{j} = data_test_struct.data;
    %% Low-pass filter based on user-defined likelihood threshold
    [MsTestingData{j},perc_rect] = dlc_preprocess(rawdata_test{j},0.2);
end
%% Classifier a test dataset that the algorithm has not seen before, no ground truth but can test against human observers
[labels,f_10fps_test] = bsoid_svm(MsTestingData,60,OF_mdl);
%% Compare with manually defined criteria
[g_num,perc_unk,trans_p] = bsoid_mt(MsTestingData,60,24);
