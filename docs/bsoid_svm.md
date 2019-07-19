## BSOID_SVM.m
**Purpose**: Predict the new dataset based on your classifier.

```matlab
function [labels,f_10fps_test] = bsoid_svm(data_test,fps,OF_mdl,smth_hstry,smth_futr)
```

### Prior to Usage
Run *[bsoid_mdl.md](bsoid_mdl.md)* to build a behavioral model, or load up our demo model.
```matlab
load OF_mdl
```
Then, 
Run *[dlc_preprocess.md](dlc_preprocess.md)* on test dataset.

```matlab
fprintf('Please select DeepLabCut generated .csv file. \n'); 
[filecsv,pathcsv] = uigetfile; 
filenamecsv = sprintf('%s%s',pathcsv,filecsv);
data_test_struct = importdata(filenamecsv); rawdata_test{i} = data_test_struct.data;
function [data_test,perc_rect] = dlc_preprocess(rawdata_test,llh)
```

#### Inputs to BSOID_SVM.m

- `DATA_TEST`    6-body parts (x,y) matrix outlining the mouse viewed from bottom-up. Rows represents frame numbers. Columns 1 & 2: Snout; Columns 3, 4, 5 & 6: two front paws (Left-Right order does not matter); Columns 7, 8, 9 & 10: two hind paws (Left-Right order does not matter); Columns 11 & 12: base of tail (Place it where the tail extends out from the butt). 

- `FPS`    Rounded video sampling frame rate. Use *VideoReader* in MATLAB or *ffmpeg* bash command to detect. 

  ```matlab
  vidObj = VideoReader('videos/thisismyvideo.mp4');
  fps = round(vidObj.FrameRate);
  ```

  ```bash
  cd videos/
  ffmpeg -i thisismyvideo.mp4
  ```

- `OF_MDL`    Percentage of data randomly held out for test. Default is 0.20 (80/20 training/testing). 

- `SMTH_HSTRY`   and `SMTH_FUTR`   designates number of frames for BOXCAR smoothing to reduce noise levels of the signal detected from DeepLabCut 2.0. This depends on your frame-rate. The default setting will automatically scale the number of frames to smooth from to approximately 40ms before and after. Obviously, the higher the sampling rate, the more de-noise this will perform.

#### Outputs of BSOID_SVM.m

- `LABELS`    Predicted action based on the model, the group here matches the group number in the bsoid_gmm, 10 frame/second temporal resolution.

- `F_10FPS_TEST`    The features collated for the test animal, 10 frame/second temporal resolution.


### Upon obtaining the outputs
**REAL SCIENCE STARTS!**
