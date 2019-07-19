## BSOID_MDL.m
**Purpose**: Build a support vector machine classifier using `f_10fps` and `grp` outputs from bsoid_gmm.m. This allows future prediction of user defined groups. The algorithm defaults at using the clusters from GMM, but user can also input the merged groups and have a classifier built to their desires. However, we would argue against building a classifier based on human perception of what is the same behavior; instead, build this classifier based on the x number of GMM clusters, and merge the predicted labels later.

```matlab
function [OF_mdl,CV_amean,CV_asem,acc_fig] = bsoid_mdl(f_10fps,grp,hldout,cv_it,btchsz)
```

### Prior to Usage

*Run [bsoid_gmm.md](bsoid_gmm.md) first*

#### Inputs to BSOID_MDL.m

- `F_10FPS`    Compiled features that were used to cluster, 10fps temporal resolution.

- `GRP`    Statistically different groups of actions based on data. Output is 10Hz.

- `HLDOUT`    Percentage of data randomly held out for test. Default is 0.20 (80/20 training/testing). 

- `CV_IT`   Number of times to run cross-validation on. Default is 100.

- `BTCHSZ`    Batch size for randsampling, make sure the hold out data is <= CV_IT*BTCHSZ. Default is 200.

- `IT`  The number of random initialization for Gaussian Mixture Models. This attempts to find global optimum, instead of local optimum. Default is 20.

#### Outputs of BSOID_MDL.m

- `OF_MDL`    Support Vector Machine Classifier Model.

- `CV_AMEAN`    Cross-validated accuracy mean.

- `CV_ASEM`    Cross-validated accuracy standard error fo the mean.

- `ACC_FIG`    Box plot showing classifier performance with individual data points representing a randomly subsampled test set from the hold out portion.

### Upon obtaining the outputs
*Run [bsoid_svm.md](bsoid_svm.md) next*
