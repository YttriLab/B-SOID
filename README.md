![Mouse Action Cluster Demo 1x](demo/Ms2ActGMMClustVidK.gif)

# B-SOID: Behavioral-Segmentation of Open-Field in DeepLabCut

B-SOID ("B-side") is an unsupervised learning algorithm written in MATLAB that analyzes sub-second rodent behavior from a single bottom-up recording video-camera. With the output trajectory of 6 body parts (snout, the 4 paws, and the base of the tial) of a rodent extracted from [DeepLabCut](https://github.com/AlexEMG/DeepLabCut)<sup>1,2</sup>, this algorithm performs t-Distributed Stochastic Neighbor Embedding (t-SNE<sup>4</sup>, MATLAB&copy;) of the 7 heirarchically different time-varying signals to fit Gaussian Mixture Models<sup>5</sup>. The output agnostically separates statistically significant distributions in the 3-dimensional action space and are found to be correlated with different observable rodent behaviors. Although the primary purpose of this integrated technology is for automatic classification of behaviors that either have rodent-to-rodent variability, or carry observer bias, the output also uncovers behaviors likely coincides better with latent variables. The idea behind this also is also expandable to multiple perspectives and other fields of study.  

## Installation

Use Git using the web URL or download ZIP. 

Change your current working directory to the location where you want the cloned directory to be made.

```bash
git clone https://github.com/YttriLab/B-SOID.git
```

## Usage
Change MATLAB current folder to `B-SOID/bsoid` 

### Step I 
Import .csv file, and convert it to a matrix
```matlab
data_struct = import(Ms2OpenField.csv);
rawdata = data_struct.data
```
### Step II
Apply a low-pass filter for data likelihood. `dlc_preprocess` finds the most recent x,y that are above the threshold and replaces with them.
```matlab
data = dlc_preprocess(rawdata,0.5);
```
### Step III
#### &nbsp;&nbsp;&nbsp;&nbsp; `Option 1`: Manual criteria for a rough but fast analyses (If you are interested in considering the rough estimate of the 7 behaviors: 1 = Pause, 2 = Rear, 3 = Groom, 4 = Sniff, 5 = Locomote, 6 = Orient Left, 7 = Orient Right)
```matlab
[g_label,g_num,perc_unk] = bsoid_fast(data,pix_cm); % data, pixel/cm
```
#### &nbsp;&nbsp;&nbsp;&nbsp; `Option 2`: Unsupervised grouping of action space, more refined and reliable output (This can uncover behaviors, *not just the 7 listed above*, and perhaps coincide better with latent variables) 

* The following steps are only valid if you go with `Option 2`
### Step IV
#### Unsupervised grouping of the purely data-driven action space
```matlab
[f,tsne_f,g_ls,g_hs,llh,bsoid_fig] = bsoid_us(data,60); % data, sampling-rate
```
### Step V (Back to the 7 states of interest)
#### Based on feature distribution of the clusters

#### Define threshold for the 7 actions (i.e. if majority of data in that unsupervised grouping has a angular change, it is turning)


### *(OPTIONAL) Step VI (If you are interested in creating short videos (.avi) of the groups). This is not recommended if it has more than 100,000 frames at 720*1280p.*
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

We are a neuroscience lab and can easily foresee this as being upgraded. Please do not hesitate in helping us working towards a more efficient and accurate algorithm. All major contributors will be cited in our publications.

## License

This software package provided without warranty of any kind and is licensed under the GNU Lesser General Public License v3.0. 
Cite us if you use the code and/or data!.https://choosealicense.com/licenses/mit/)

## References
1. [Mathis A, Mamidanna P, Cury KM, Abe T, Murthy VN, Mathis MW, Bethge M. DeepLabCut: markerless pose estimation of user-defined body parts with deep learning. Nat Neurosci. 2018 Sep;21(9):1281-1289. doi: 10.1038/s41593-018-0209-y. Epub 2018 Aug 20. PubMed PMID: 30127430.](https://www.nature.com/articles/s41593-018-0209-y)

2. [Nath T, Mathis A, Chen AC, Patel A, Bethge M, Mathis MW. Using DeepLabCut for 3D markerless pose estimation across species and behaviors. Nat Protoc. 2019 Jul;14(7):2152-2176. doi: 10.1038/s41596-019-0176-0. Epub 2019 Jun 21. PubMed PMID: 31227823.](https://doi.org/10.1038/s41596-019-0176-0)

3. [Insafutdinov E., Pishchulin L., Andres B., Andriluka M., Schiele B. (2016) DeeperCut: A Deeper, Stronger, and Faster Multi-person Pose Estimation Model. In: Leibe B., Matas J., Sebe N., Welling M. (eds) Computer Vision – ECCV 2016. ECCV 2016. Lecture Notes in Computer Science, vol 9910. Springer, Cham](http://arxiv.org/abs/1605.03170)

4. [L.J.P. van der Maaten. Accelerating t-SNE using Tree-Based Algorithms. Journal of Machine Learning Research 15(Oct):3221-3245, 2014.](https://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf)

5. [Chen M. EM Algorithm for Gaussian Mixture Model (EM GMM). MATLAB Central File Exchange. Retrieved July 15, 2019.](https://www.mathworks.com/matlabcentral/fileexchange/26184-em-algorithm-for-gaussian-mixture-model-em-gmm)
