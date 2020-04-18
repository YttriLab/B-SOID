## Python3 Tutorial (not compatible with Python 2.)

## Install the necessary utilities 
### Step 1: Install Anaconda/Python3
[Anaconda](https://www.anaconda.com/) is a free and open source distribution of the Python programming language. 
Make sure you select python3.
![Step1](demo/py3_step1.gif)
### Step 2: Create virtual environment for your B-SOiD project
```
conda create -n bsoid_env
conda activate bsoid_env
```
![Step2](demo/py3_step2.gif)
You should now see (bsoid_env) $yourusername@yourmachine ~ %

### Step 3: install dependencies
```
conda install ipython  
pip install pandas tqdm matplotlib opencv-python seaborn scikit-learn
```
![Step3](demo/py3_step3.gif)

### Step 4: Clone B-SOID and change your current directory to B-SOID/bsoid_py/config to edit your configuration path
```
git clone https://github.com/YttriLab/B-SOID.git
cd B-SOID/bsoid_py/config/
vim LOCAL_CONFIG.py
```
![Step4](demo/py3_step4.gif)

Use the vim commands as follows:
* vim commands are:
    * "i" for going into edit mode, user keyboards to navigate.
    * "esc" for escaping editor mode. 
    * ":w" write changes.
    * ":wq" for writing your changes and quit vim.
    * ":q!" disregarding any changes and quit.

The things to change here are:
* BASE_PATH = '/Users/ahsu/B-SOID/datasets'
    * Change this to your directory to the project folder that all experimental folders nests within.
* TRAIN_FOLDERS = ['/Train1', '/Train2']
    * Within BASE_PATH, you now list the immediate parent folder that houses .csv files.
* PREDICT_FOLDERS = ['/041919', '/042219']
    * Folder paths containing new dataset (.csv files) to predict using built classifier.
* BODYPARTS = {
    'Snout/Head': 0,
    'Neck': None,
    'Forepaw/Shoulder1': 1,
    'Forepaw/Shoulder2': 2,
    'Bodycenter': None,
    'Hindpaw/Hip1': 3,
    'Hindpaw/Hip2': 4,
    'Tailbase': 5,
    'Tailroot': None
}
    * Change the order if its different. 
    * Please make sure you have the 6 that I highlighted here. 
    * You can have more, but not less.
* FPS = 60
    * Change this to your camera frame-rate. 
    * Determine that by getting info from your videos. 
    * Be as precise as possible.
* OUTPUT_PATH = '/Users/ahsu/Desktop/'
    * Change to desired output path that the program will be saving results to.
* MODEL_NAME = 'c57bl6_n2_120min'
    * If you have multiple different experimental conditions and/or would like to play with the machine learning parameters, you can name the machine learning model.
* FINALMODEL_NAME = 'bsoid_c57bl6_n2_120min_20200413_1941.sav'
    * Once you're comfortable with your model, you can set the final model designation.
    * This is calling the name of the model you'll be using for any future predictions.
* VID_NAME = '/Users/ahsu/B-SOID/datasets/041919/2019-04-19_09-34-36cut0_30min.mp4'
    * If you want to visualize the output to understand what each group means, pick a video.
* ID = 0
    * Which csv ID number would this video be corresponding to?
        * For example, if this is the first video in the first PREDICT_FOLDERS list: /041919, it will be 0. (0 index; 1-1)
        * If there are only 2 videos per folder, and it's the second video in the second PREDICT_FOLDERS list: /042219, it will be 3 (0 index; 4-1)
* FRAME_DIR = '/Users/ahsu/B-SOID/datasets/041919/0_30min_10fpsPNGs'
    * Where would you want the extracted frames to be stored?
    * Make sure you've created this folder and it exists!
* SHORTVID_DIR = '/Users/ahsu/B-SOID/datasets/041919/examples'
    * Where would these short example videos be saved to?
    * Make sure you've created this folder and it exists!

### Step 4.1 (OPTIONAL): Change plotting/video configuration.
Within the same B-SOID/bsoid_py/config/ directory
```
vim GLOBAL_CONFIG.py
```
![Step4_1](demo/py3_step4_1.gif)
* PLOT_TRAINING = True
    * Change to False if you don't want plots. It'll still save the output .csvs.
* GEN_VIDEOS = True
    * Change to False if you don't want video outputs during the prediction process.
    * Generally, I would only keep this True to validate the model as video takes time to process.
    
You can also tweak other parameters in the file, but it's not necessary.
    
### Step 5: Build your own machine learning behavioral classifier!
Go back to main B-SOID folder, if you're in bsoid_py, `cd ..`
First, call iPython
```
ipython
```
Then import configuration files, and the main function
```
from bsoid_py.config import *
import bsoid_py.main
```
Build your behavioral model using your training csv files within your train folders (LOCAL_CONFIG.py):
```
bsoid_py.main.build(TRAIN_FOLDERS)
```
![Step5](demo/py3_step5.gif)
Note that this process takes some time depending on how big your dataset is. 
Generally, I would consider training with at least 3 animals, with sufficient behavioral data.

Once this process is done

### Step 6: Run your classifier on all future datasets!
Still inside iPython, see above.
```
bsoid_py.main.run(PREDICT_FOLDERS)
```
![Step6](demo/py3_step6.gif)
The output results will be stored automatically in the OUTPUT_PATH (LOCAL_CONFIG.py).
Video results will be stored as well (LOCAL_CONFIG.py).

### VOILA!

