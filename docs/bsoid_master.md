## Detailing steps for the master script `bsoid_master_v1p2.m`

### Part 1
Import your .csv file from DeepLabCut, and convert it to a matrix. We can run a for loop to get all of the .csv loaded into your matlab environment. After each individual raw .csv files are imported as a matrix, we evaluate the likelihoods of all points and apply a high-pass filter for estimated data likelihood `adp_filt`.
```matlab
n = 2; % How many .csv files do you want to build your model on?
for i = 1:n
    %% Import data
    fprintf(sprintf('%s%s%s','Please select ',num2str(i),' DeepLabCut generated .csv file for training data. \n'));  
    cd /your/dlc/output/folder/ % make sure you change this to the folder you have saved the .csv files for pose estimation
    [filecsv,pathcsv] = uigetfile('*.csv'); % shows you only .csv files
    filenamecsv = sprintf('%s%s',pathcsv,filecsv); 
    data_struct = importdata(filenamecsv); 
    rawdata{i} = data_struct.data; % import data as matrix in a cell
    %% Adaptive high-pass filter based on data for parts that are easily occluded
    [MsTrainingData{i},perc_rect] = adp_filt(rawdata{i});
end
```

Alternatively, load the Yttri lab's demo training dataset.
```matlab
load MsTrainingData.mat
```

### Part 2
#### &nbsp;&nbsp;&nbsp;&nbsp; `Option 1`: Unsupervised grouping of the purely data-driven t-SNE space based on Gaussian Mixture Models (GMM). Refer to [bsoid_gmm.md](docs/bsoid_gmm.md). With version 1.2, try bsoid_assign.m for better separation of clusters, especially for larger datasets.

```matlab
%% Segment the groups based on natural statistics of input features, refer to our paper for feature list
[feats,tsne_feats,grp,llh,bsoid_fig] = bsoid_assign(MsTrainingData,FPS,COMP); % Change 60 to your frame rate. Set 1 for a compiled space for all .csv.
```

Alternatively, you can load the demo f_10fps and groupings.
```matlab
load MsTrainingFeats.mat MsActionGrps.mat
```
![3D Action Space Groups Demo 1x](demo/3DActionSpaceGrps1.gif).

The 3-dimensional figure above shows the agnostic groupings of our demo training dataset undergoing unsupervised learning classification. 

### Part 3
#### Build a personalized multi-class Support Vector Machine (SVM) classifier based on feature distribution of the individual GMM groups. Refer to [bsoid_mdl.md](docs/bsoid_mdl.md). With version 1.2, bsoid_mdl2.m computes the size of each group to run cross-validation for a given iteration and train/test ratio. For example, if you desire a 80/20 = train/test, and that your dataset is 1000 behavioral data points, with 10 cross-validated iterations, the size of each test group will be 20 behavioral data points, `1000*0.2/10 = 20`. 
```matlab
%% Build a Support Vector Machine (SVM) classifier based on your data
[behv_mdl,cv_amean] = bsoid_mdl2(feats,grp,0.2,10); % Change 0.2 to desired ratio of held out data to test the classifier on. 
% Save your model as a .mat file if it looks good.
```
If you are interested in using our model,
```matlab
load OF_mdl
```
![Model performance](demo/MsTrainingSVM_Accuracy.png)
The figure above shows SVM model performance on 20% of the data that was held out from training. Each dot represents 200 randomly sampled actions, and there are 70 different iterations, without replacement, for showing the robust cross-validation accuracy.

### Part 4
#### With the model built, we can accurately and quickly predict future mouse datasets based on DeepLabCut predicted pose. Refer to [bsoid_svm.md](docs/bsoid_svm.md). 

```matlab
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
[labels,f_10fps_test] = bsoid_svm(MsTestingData,FPS,behv_mdl); % Change 60 to your frame rate.   
```

You can attempt to test this on our demo test dataset
```matlab
load MsTestingData.mat
[labels,f_10fps_test] = bsoid_svm(MsTestingData,OF_mdl);
```

### Part 5
#### Using the classifier, we can utilize frame-shift paradigm to extract behaviors for every single frame based on DeepLabCut predicted pose.
```matlab
%% In addition, you can play with frame-shifted machine learning prediction for detection of behavioral start up to camera frame rate
[labels_fsALL,f_10fps_fs] = bsoid_fsml(MsTestingData,FPS,behv_mdl); % Change 60 to your frame rate.
```


#### &nbsp;&nbsp;&nbsp;&nbsp; `Option 2`: Manual criteria for a rough but fast analysis (If you are interested in considering the rough estimate of the 7 behaviors: 1 = Pause, 2 = Rear, 3 = Groom, 4 = Sniff, 5 = Locomote, 6 = Orient Left, 7 = Orient Right). Refer to [bsoid_mt.md](docs/bsoid_mt.md)
Based on our zoom from the 15 inch x 12 inch open field set-up, at a camera resolution of 1280p x 720p, we have set criteria for the 7 states of action. This fast algorithm was able to automatically detect the gross behavioral changes in a Parkisonian mouse model. This can serve as a quick first pass at analyzing biases in transition matrices and overarching behavioral changes before digging further into the behavior (`Option2`).
```matlab
[g_label,g_num,perc_unk] = bsoid_mt(data,pix_cm); % data, pixel/cm
```
If you are using our demo dataset
```matlab
load MsTestingData.mat
[g_label,g_num,perc_unk] = bsoid_mt(MsTestingData,24); % data, pixel/cm
```