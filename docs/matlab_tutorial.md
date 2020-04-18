## MATLAB Tutorial

Change the MATLAB current folder to the folder containing `B-SOID/bsoid` 

### Setting things up
If you are interested in creating short videos (.avi) of the groups to help users subjectively define the various actions.
#### Install [FFmpeg](https://github.com/adaptlearning/adapt_authoring/wiki/Installing-FFmpeg) or other software that can achieve the same thing, I will provide the FFmpeg command lines below

Go to your video directory.
```C
cd your/video/directory/mouse.mp4
```

#### Use FFmpeg to create 10fps frames from your videos to match grp
```C
ffmpeg -i your_highspeedvideo.mp4 -filter:v fps=fps=10 your_10fpsvideo.mp4
mkdir your_10fpsPNG
ffmpeg -i "your_10fpsvideo.mp4" your_10fpsPNG/img%01d.png
```

Keep track of which video you have extracted frames from. I would use that as the first .csv to be selected once you run the master script `bsoid_master_v1p2.m`. To do so, set `s_no = 1` in line 32 of the master script. This is calling the first .csv input and map behavioral indices to frames you have extracted from the video.  

### The only step you need, which is setting your parameters!
In MATLAB, open `bsoid_master_v1p2.m`. 
Now set up parameters for the master script.

```matlab
%%% bsoid_master_v1p2 line 7-8
% Set framerate and classifier count
FPS = 60;
COMP = 1; % 1 classifier, 0 for individual classifier/csv
```

In addition, let's set the number of .csv files you have to build a classifier on. For instance, if you have 3 .csv files generated from DeepLabCut, set `n = 3` in line 8 of the master script. This will compile the data into a cell of matrices. If, however, you have only 1 .csv file you want to test this program out, you will still need to run the for loop, set `n = 1` in line 8, for proper data format.
```matlab
%%% bsoid_master_v1p2 line 11
n = 3; % How many .csv files do you want to build your model on?
```

Lastly, before we run the master script, let's set the number of .csv files you want to predict behaviors based on pose using your own SVM model that you just trained. If you want to predict the same .csv files with the machine learning model, set `m = 3` or `m = 1` in line 36 of the master script.
```matlab
%%% bsoid_master_v1p2 line 30
%% Once you trained your action model
m = 3; % How many .csv do you want to test on?
```

What was the video you extracted frames from? set s_no = m (for that csv/video frames)
```matlab
%%% bsoid_master_v1p2 line 51
% Assuming you trained on multiple sessions, select the session number corresponding to your video/frames
s_no = 1;
```


Once this is all done, click run.

Note that it will pop up user interfaces for you to select the files. Make sure you follow the printed statements for what to be selected.
`Please select 2 DeepLabCut generated .csv file for training data.` is asking you to select the second training .csv files.
`Please select the folder containing FFmpeg generated frames from your 10fps video.` is asking you to select the folder containing extracted frames.
`Please select output folder for GIF.` is asking you to select/create an output folder for snippets of extracted behaviors (.avi) 
`Please select 1 DeepLabCut generated .csv file for testing data.` is asking you to select the first testing .csv files.


Alternatively, you can learn more about the [algorithm](bsoid_master.md) and only adopt one or few of the following steps. 
