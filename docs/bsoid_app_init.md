## B-SOID APP initialization

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
pip install pandas psutil tqdm matplotlib opencv-python ffmpeg-python seaborn scikit-learn networkx umap-learn hdbscan streamlit
```

### Step 4: Clone B-SOID
```
git clone https://github.com/YttriLab/B-SOID.git
```

### Step 5: Run the App! It'll open up in your default browser.
```
cd B-SOID
streamlit run bsoid_app.py
```
