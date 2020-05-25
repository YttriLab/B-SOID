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
pip install pandas tqdm matplotlib opencv-python ffmpeg-python seaborn scikit-learn umap-learn hdbscan streamlit
```

### Step 4: Clone B-SOID and change your current directory to B-SOID/bsoid_py/config to edit your configuration path
```
git clone https://github.com/YttriLab/B-SOID.git
```

### Step 5: Run the App! It'll open up in your default browser.
```
cd B-SOID
streamlit run bsoid_app.py
```
