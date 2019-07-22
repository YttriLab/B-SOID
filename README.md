# B-SOID: Behavioral segmentation of open field in DeepLabCut

[DeepLabCut](https://github.com/AlexEMG/DeepLabCut)<sup>1,2,3</sup> has revolutionized the way behavioral scientists analyze data. The algorithm utilizes recent advances in computer vision and deep learning to automatically estimate 3D-poses. Interpreting the positions of an animal can be useful in studying behavior; however, it does not encompass the whole dynamic range of naturalistic behaviors. 

Behavioral segmentation of open field in DeepLabCut, or B-SOID ("B-side"), is an unsupervised learning algorithm written in MATLAB that serves to discover behaviors that are not pre-defined by users. Our algorithm can segregate statistically different sub-second rodent behaviors with a single bottom-up perspective video-camera. Upon DeepLabCut estimating the positions of 6 body parts (snout, the 4 paws, and the base of the tial) outlining a rodent navigating an open environment, this algorithm performs t-Distributed Stochastic Neighbor Embedding (t-SNE<sup>4</sup>, MATLAB&copy;) of the 7 different time-varying signals to fit Gaussian Mixture Models<sup>5</sup>. The output agnostically separates statistically significant distributions in the 3-dimensional action space and are found to be correlated with different observable rodent behaviors.

This usage of this algorithm has been outlined below, and is extremely flexible in adapting to what the user wants. With the ever-blooming advances in ways to study an animal behavior, our algorithm builds on and integrates what has already been robustly tested to help advance scientific research.

![Mouse Action Cluster Demo 1x](demo/2x2grid.gif)

The dataset from *Yttri lab, Alexander Hsu,* (left) has been tested against multiple human observers and showed comparable inter-grader variability as another observer. We also tested the generalizability with the dataset from *Ahamri lab , Jared Kopelman, Shirley Jiang, & Sean Piantadosi* (right), and was predictive of actual behavior.

## Installation

Git clone the web URL or download ZIP. 

Change your current working directory to the location where you want the cloned directory to be made.

```bash
git clone https://github.com/YttriLab/B-SOID.git
```

## Usage
Change the MATLAB current folder to the folder containing `B-SOID/bsoid` 

### Step I 
Import your .csv file from DeepLabCut, and convert it to a matrix.
```matlab
data_struct = import(your_DLC_output.csv);
rawdata = data_struct.data
```
### Step II
Apply a low-pass filter for data likelihood. `dlc_preprocess` replaces drop data points with the most recent position. Refer to [dlc_preprocess.md](docs/dlc_preprocess.md).
Based on our pixel-error, the default has been set to 0.2.
```matlab
data = dlc_preprocess(rawdata,0.2);
```
Alternatively, load the Yttri lab's demo training dataset.
```matlab
load MsTrainingData.mat
```
### Step III
#### &nbsp;&nbsp;&nbsp;&nbsp; `Option 1`: Manual criteria for a rough but fast analysis (If you are interested in considering the rough estimate of the 7 behaviors: 1 = Pause, 2 = Rear, 3 = Groom, 4 = Sniff, 5 = Locomote, 6 = Orient Left, 7 = Orient Right). Refer to [bsoid_mt.md](docs/bsoid_mt.md)
Based on our zoom from the 15 inch x 12 inch open field set-up, at a camera resolution of 1280p x 720p, we have set criteria for the 7 states of action. This fast algorithm was able to automatically detect the gross behavioral changes in a Parkisonian mouse model. This can serve as a quick first pass at analyzing biases in transition matrices and overarching behavioral changes before digging further into the behavior (`Option2`).
```matlab
[g_label,g_num,perc_unk] = bsoid_mt(data,pix_cm); % data, pixel/cm
```
If you are using our demo dataset
```matlab
load MsTestingData.mat
[g_label,g_num,perc_unk] = bsoid_mt(MsTestingData,24); % data, pixel/cm
```
#### &nbsp;&nbsp;&nbsp;&nbsp; `Option 2`: Unsupervised grouping of the purely data-driven action space based on Gaussian Mixture Models (GMM). Refer to [bsoid_gmm.md](docs/bsoid_gmm.md)
```matlab
[feats,f_10fps,tsne_feats,grp,llh,bsoid_fig] = bsoid_gmm(data,fps,1); % data, frame rate, 1 classifier for all.
```
Alternatively, you can load the demo f_10fps and groupings.
```matlab
load MsTrainingFeats.mat MsActionGrps.mat
```
![3D Action Space Groups Demo 1x](demo/3DActionSpaceGrps.gif)
The 3-dimensional figure above shows the agnostic groupings of our demo training dataset undergoing unsupervised learning classification. 

## The following steps are only applicable if you go with `Option 2`
### Step IV 
#### Build a personalized Support Vector Machine (SVM) classifier based on feature distribution of the individual GMM groups. Refer to [bsoid_mdl.md](docs/bsoid_mdl.md).
```matlab
[OF_mdl,CV_amean,CV_asem,acc_fig] = bsoid_mdl(f_10fps,grp); % features and GMM groups from bsoid_gmm
```
If you are interested in using our model,
```matlab
load OF_mdl
```
![Model performance](demo/MsTrainingSVM_Accuracy.png)
The figure above shows SVM model performance on 20% of the data that was held out from training. Each dot represents 200 randomly sampled actions, and there are 70 different iterations, without replacement, for showing the robust cross-validation accuracy.


### Step V
#### With the model built, we can accurately and quickly predict future mouse datasets by just looking at their features. This is essentially `Option 1`, but based on machine learning. Refer to [bsoid_svm.md](docs/bsoid_svm.md)

```matlab
data_test_struct = import(new_mouse.csv);
rawdata_test = data_test_struct.data
data_test = dlc_preprocess(rawdata_test,0.1);
[labels,f_10fps_test] = bsoid_svm(data_test,OF_mdl); % features and GMM groups from bsoid_gmm
```
You can attempt to test this on our demo test dataset
```matlab
load MsTestingData.mat
[labels,f_10fps_test] = bsoid_svm(MsTestingData,OF_mdl);
```

### *(OPTIONAL) Step VI (If you are interested in creating short videos (.avi) of the groups to help users subjectively define the various actions).*
#### Read the video and create a handle for it.
```matlab
vidObj = VideoReader(filenamevid); % video used to generate DLC
```
#### Assuming all behaviors can be sampled from the first 10 minutes (600 seconds), have MATLAB store only every 10 frames per second.
```matlab
k = 1; kk = 1;
while vidObj.CurrentTime < 599.9
    vidObj.CurrentTime = k/60;
    video{i}(kk).cdata = readFrame(vidObj); % save only every 10fps
    kk = kk+1; % save only every 10fps
    k = k+round(vidObj.FrameRate)/10; % reduce down to 10fps (100ms/frm)
end
```
#### Create short videos in the desired output folder (default = current directory) of different groups of action clusters that at least lasted for ~300ms, and slow the video down to 0.75X for better understanding.
```matlab
filepathout = uigetdir;
[t,b,b_ex] = action_gif(video,grp_fill,1,5,6,0.75,filepathout);
```


## Contributing

Pull requests are welcome. For recommended changes that you would like to see, open an issue.

We are a neuroscience lab and welcome all contributions to improve this algorithm. Please do not hesitate to contact us for any question/suggestion.

## License

This software package provided without warranty of any kind and is licensed under the GNU Lesser General Public License v3.0. 
Cite us if you use the code and/or data! (https://choosealicense.com/licenses/mit/)

## References
1. [Mathis A, Mamidanna P, Cury KM, Abe T, Murthy VN, Mathis MW, Bethge M. DeepLabCut: markerless pose estimation of user-defined body parts with deep learning. Nat Neurosci. 2018 Sep;21(9):1281-1289. doi: 10.1038/s41593-018-0209-y. Epub 2018 Aug 20. PubMed PMID: 30127430.](https://www.nature.com/articles/s41593-018-0209-y)

2. [Nath T, Mathis A, Chen AC, Patel A, Bethge M, Mathis MW. Using DeepLabCut for 3D markerless pose estimation across species and behaviors. Nat Protoc. 2019 Jul;14(7):2152-2176. doi: 10.1038/s41596-019-0176-0. Epub 2019 Jun 21. PubMed PMID: 31227823.](https://doi.org/10.1038/s41596-019-0176-0)

3. [Insafutdinov E., Pishchulin L., Andres B., Andriluka M., Schiele B. (2016) DeeperCut: A Deeper, Stronger, and Faster Multi-person Pose Estimation Model. In: Leibe B., Matas J., Sebe N., Welling M. (eds) Computer Vision – ECCV 2016. ECCV 2016. Lecture Notes in Computer Science, vol 9910. Springer, Cham](http://arxiv.org/abs/1605.03170)

4. [L.J.P. van der Maaten. Accelerating t-SNE using Tree-Based Algorithms. Journal of Machine Learning Research 15(Oct):3221-3245, 2014.](https://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf)

5. [Chen M. EM Algorithm for Gaussian Mixture Model (EM GMM). MATLAB Central File Exchange. Retrieved July 15, 2019.](https://www.mathworks.com/matlabcentral/fileexchange/26184-em-algorithm-for-gaussian-mixture-model-em-gmm)
