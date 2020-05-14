## Python3 v1.3 (UMAP + HDBSCAN) Tutorial (not compatible with Python 2.)

## Install the necessary utilities 
### Step 1: Install Anaconda/Python3
[Anaconda](https://www.anaconda.com/) is a free and open source distribution of the Python programming language. 
Make sure you select python3.
### Step 2: Create virtual environment for your B-SOiD project
```
conda create -n bsoid_env
conda activate bsoid_env
```
You should now see (bsoid_env) $yourusername@yourmachine ~ %

### Step 3: install dependencies
```
conda install ipython  
pip install pandas tqdm matplotlib opencv-python seaborn scikit-learn umap-learn hdbscan
```

### Step 4: Clone B-SOID and change your current directory to B-SOID/bsoid_py/config to edit your configuration path
```
git clone https://github.com/YttriLab/B-SOID.git
cd B-SOID/bsoid_umap/config/
vim LOCAL_CONFIG.py
```

Use the vim commands as follows:
* vim commands are:
    * "i" for going into edit mode, user keyboards to navigate.
    * "esc" for escaping editor mode. 
    * ":w" write changes.
    * ":wq" for writing your changes and quit vim.
    * ":q!" disregarding any changes and quit.

If you are a windows user and does not have vim installed, it is recommended to install python text editors to edit the LOCAL_CONFIG.py

ATOM text editor: https://atom.io/

#### The things to change here are:
* BASE_PATH = '/Users/ahsu/B-SOID/datasets'
    * Change this to your directory to the project folder that all experimental folders nests within.
* TRAIN_FOLDERS = ['/Train1', '/Train2']
    * Within BASE_PATH, you now list the immediate parent folder that houses .csv files.
* PREDICT_FOLDERS = ['/041919', '/042219']
    * Folder paths containing new dataset (.csv files) to predict using built classifier.
* FPS = 60
    * Change this to your camera frame-rate. 
    * Determine that by getting info from your videos. 
    * Be as precise as possible.
* OUTPUT_PATH = '/Users/ahsu/Desktop/'
    * Change to desired output path that the program will be saving results to.
* MODEL_NAME = 'c57bl6_n2_120min'
    * Name the machine learning model.
* PLOT = True
    * Change to False if you don't want plots. It'll still save the output .csvs.
    
* VID = True
    * Change to False if you don't want video outputs during the prediction process.
    * Generally, I would only keep this True to validate the model as video takes time to process.
    * If True, make sure direct to the video below AND that you created the two specified folders!
* FRAME_DIR = '/Users/ahsu/B-SOID/datasets/041919/0_30min_10fpsPNGs'
    * Where would you want the extracted frames to be stored?
    * Make sure you've created this folder and it exists!
* SHORTVID_DIR = '/Users/ahsu/B-SOID/datasets/041919/examples'
    * Where would these short example videos be saved to?
    * Make sure you've created this folder and it exists!
* VID_NAME = '/Users/ahsu/B-SOID/datasets/041919/2019-04-19_09-34-36cut0_30min.mp4'
    * If you want to visualize the output to understand what each group means, pick a video.
* ID = 0
    * Which csv ID number would this video be corresponding to?
        * For example, if this is the first video in the first PREDICT_FOLDERS list: /041919, it will be 0. (0 index; 1-1)
        * If there are only 2 videos per folder, and it's the second video in the second PREDICT_FOLDERS list: /042219, it will be 3 (0 index; 4-1)

### Step 5: Build your own machine learning behavioral classifier!
Go back to main B-SOID folder, if you're in bsoid_py, `cd ..`
First, call iPython
```
ipython
```
Then import configuration files, and the main function
```
from bsoid_umap.config import *
import bsoid_umap.main
```
Build your behaviaoral model using your training csv files within your train folders (LOCAL_CONFIG.py):
```
f_10fps, f_10fps_sc, umap_embeddings, hdb_assignments, soft_assignments, soft_clusters, nn_classifier, scores, nn_assignments = bsoid_umap.main.build(TRAIN_FOLDERS)
```
Note that this process takes some time depending on how big your dataset is. 
Generally, I would consider training with at least 3 animals, with sufficient behavioral data.

### Step 6: Run your classifier on all future datasets!
Still inside iPython, see above.
```
data_new, fs_labels = bsoid_umap.main.run(PREDICT_FOLDERS)
```
The output results will be stored automatically in the OUTPUT_PATH (LOCAL_CONFIG.py).
Video results will be stored as well (LOCAL_CONFIG.py).

### VOILA!

