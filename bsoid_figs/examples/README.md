
## [Pose relationships histograms](../github_hist.py)
`python github_hist.py` 

Runs the following subroutines

### BOTTOM UP CAMERA -- 11 GROUPS  
<p align="center">
  INACTIVE
</p>

![inactive 0.25x](behavioral_videos/inactive_hstacked.gif)

<p align="center">
  INVESTIGATE
</p>

![investigate 0.25x](behavioral_videos/investigate_hstacked.gif)

<p align="center">
  REAR (-)
</p>

![rear (-) 0.25x](behavioral_videos/rear_minus_hstacked.gif)

<p align="center">
  REAR (+)
</p>

![rear (+) 0.25x](behavioral_videos/rear_plus_hstacked.gif)

<p align="center">
  FACE GROOM
</p>

![face groom 0.25x](behavioral_videos/face_groom_hstacked.gif)

<p align="center">
  HEAD GROOM
</p>

![head groom 0.25x](behavioral_videos/head_groom_hstacked.gif)

<p align="center">
  BODY LICK
</p>

![body lick 0.25x](behavioral_videos/body_groom_hstacked.gif)

<p align="center">
  ITCH
</p>

![itch 0.25x](behavioral_videos/itch_hstacked.gif)

<p align="center">
  ORIENT LEFT
</p>

![orient left 0.25x](behavioral_videos/orient_L_hstacked.gif)

<p align="center">
  ORIENT RIGHT
</p>

![orient right 0.25x](behavioral_videos/orient_R_hstacked.gif)

<p align="center">
  LOCOMOTE
</p>

![locomote 0.25x](behavioral_videos/locomote_hstacked.gif)






* [Plots pose relationships bottom-up](../subroutines/pose_relationships_hist.py) for each segmented behavior.
This also converts pixels to centimeters with a scale of 23.5126 pixels/cm

`../subroutines/pose_relationships_hist.py -p, path, -f, name,
                      -r, order, -m, fig_format, -o, outpath`

             
##### Any two pair distances bottom up camera
##### Points 1 to 6 are: Snout; right forepaw; left forepaw; right hindpaw; left hindpaw; tail-base
##### Each row is representing each behavior, ordered as above gifs, with the first row being the
##### remaining non-clusterable noise data points that the random forest classifier was not trained on. 

<p align="center">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships/['distance between points:', 1, 2]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships/['distance between points:', 1, 3]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships/['distance between points:', 1, 4]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships/['distance between points:', 1, 5]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships/['distance between points:', 1, 6]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships/['distance between points:', 2, 3]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships/['distance between points:', 2, 4]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships/['distance between points:', 2, 5]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships/['distance between points:', 2, 6]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships/['distance between points:', 3, 4]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships/['distance between points:', 3, 5]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships/['distance between points:', 3, 6]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships/['distance between points:', 4, 5]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships/['distance between points:', 4, 6]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships/['distance between points:', 5, 6]_histogram.png" width="300">
</p>

##### Angular change in any two pair distances over time
##### Points 1 to 6 are: Snout; right forepaw; left forepaw; right hindpaw; left hindpaw; tail-base
##### Each row is representing each behavior, ordered as above gifs, with the first row being the
##### remaining non-clusterable noise data points that the random forest classifier was not trained on. 
<p align="center">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships/['angular change for points:', 1, 2]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships/['angular change for points:', 1, 3]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships/['angular change for points:', 1, 4]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships/['angular change for points:', 1, 5]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships/['angular change for points:', 1, 6]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships/['angular change for points:', 2, 3]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships/['angular change for points:', 2, 4]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships/['angular change for points:', 2, 5]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships/['angular change for points:', 2, 6]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships/['angular change for points:', 3, 4]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships/['angular change for points:', 3, 5]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships/['angular change for points:', 3, 6]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships/['angular change for points:', 4, 5]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships/['angular change for points:', 4, 6]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships/['angular change for points:', 5, 6]_histogram.png" width="300">
</p>

##### Displacement in single pose over time
##### Points 1 to 6 are: Snout; right forepaw; left forepaw; right hindpaw; left hindpaw; tail-base
##### Each row is representing each behavior, ordered as above gifs, with the first row being the
##### remaining non-clusterable noise data points that the random forest classifier was not trained on. 
<p align="center">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships/['displacement for point:', 1, 1]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships/['displacement for point:', 2, 2]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships/['displacement for point:', 3, 3]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships/['displacement for point:', 4, 4]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships/['displacement for point:', 5, 5]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships/['displacement for point:', 6, 6]_histogram.png" width="300">

</p>

### TOP DOWN CAMERA -- 8 GROUPS     
* [Plots pose relationships top-down](../subroutines/pose_relationships_hist2.py) for each segmented behavior.
This also converts pixels to centimeters with a scale of 14.7553 pixels/cm

`../subroutines/pose_relationships_hist2.py -p, path, -f, name,
                      -r, order, -m, fig_format, -o, outpath`
                      
##### Any two pair distances top down camera
##### Points 1 to 6 are: Snout; right shoulder; left shoulder; right hip; left hip; tail-base
##### Each row is representing each behavior, specified in manuscript, with the first row being the 
##### remaining non-clusterable noise data points that the random forest classifier was not trained on.  
<p align="center">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships_topdown/openfield_60min_N1_['distance between points:', 1, 2]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships_topdown/openfield_60min_N1_['distance between points:', 1, 3]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships_topdown/openfield_60min_N1_['distance between points:', 1, 4]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships_topdown/openfield_60min_N1_['distance between points:', 1, 5]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships_topdown/openfield_60min_N1_['distance between points:', 1, 6]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships_topdown/openfield_60min_N1_['distance between points:', 2, 3]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships_topdown/openfield_60min_N1_['distance between points:', 2, 4]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships_topdown/openfield_60min_N1_['distance between points:', 2, 5]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships_topdown/openfield_60min_N1_['distance between points:', 2, 6]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships_topdown/openfield_60min_N1_['distance between points:', 3, 4]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships_topdown/openfield_60min_N1_['distance between points:', 3, 5]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships_topdown/openfield_60min_N1_['distance between points:', 3, 6]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships_topdown/openfield_60min_N1_['distance between points:', 4, 5]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships_topdown/openfield_60min_N1_['distance between points:', 4, 6]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships_topdown/openfield_60min_N1_['distance between points:', 5, 6]_histogram.png" width="300">
</p>

##### Angular change in any two pair distances over time
##### Points 1 to 6 are: Snout; right shoulder; left shoulder; right hip; left hip; tail-base
##### Each row is representing each behavior, specified in manuscript, with the first row being the 
##### remaining non-clusterable noise data points that the random forest classifier was not trained on.  
<p align="center">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships_topdown/openfield_60min_N1_['angular change for points:', 1, 2]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships_topdown/openfield_60min_N1_['angular change for points:', 1, 3]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships_topdown/openfield_60min_N1_['angular change for points:', 1, 4]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships_topdown/openfield_60min_N1_['angular change for points:', 1, 5]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships_topdown/openfield_60min_N1_['angular change for points:', 1, 6]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships_topdown/openfield_60min_N1_['angular change for points:', 2, 3]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships_topdown/openfield_60min_N1_['angular change for points:', 2, 4]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships_topdown/openfield_60min_N1_['angular change for points:', 2, 5]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships_topdown/openfield_60min_N1_['angular change for points:', 2, 6]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships_topdown/openfield_60min_N1_['angular change for points:', 3, 4]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships_topdown/openfield_60min_N1_['angular change for points:', 3, 5]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships_topdown/openfield_60min_N1_['angular change for points:', 3, 6]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships_topdown/openfield_60min_N1_['angular change for points:', 4, 5]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships_topdown/openfield_60min_N1_['angular change for points:', 4, 6]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships_topdown/openfield_60min_N1_['angular change for points:', 5, 6]_histogram.png" width="300">
</p>

##### Displacement in single pose over time
##### Points 1 to 6 are: Snout; right shoulder; left shoulder; right hip; left hip; tail-base
##### Each row is representing each behavior, specified in manuscript, with the first row being the 
##### remaining non-clusterable noise data points that the random forest classifier was not trained on.  
<p align="center">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships_topdown/openfield_60min_N1_['displacement for point:', 1, 1]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships_topdown/openfield_60min_N1_['displacement for point:', 2, 2]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships_topdown/openfield_60min_N1_['displacement for point:', 3, 3]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships_topdown/openfield_60min_N1_['displacement for point:', 4, 4]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships_topdown/openfield_60min_N1_['displacement for point:', 5, 5]_histogram.png" width="300">
  <img src="https://github.com/runninghsus/bsoid_figs/blob/main/examples/pose_relationships_topdown/openfield_60min_N1_['displacement for point:', 6, 6]_histogram.png" width="300">

</p>
