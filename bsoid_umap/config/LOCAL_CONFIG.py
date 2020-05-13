################### THINGS YOU MAY WANT TO CHANGE ###################

BASE_PATH = '/Users/ahsu/B-SOID/datasets'  # Base directory path.
TRAIN_FOLDERS = ['/Train1', '/Train2']  # Data folders used to training neural network.
PREDICT_FOLDERS = ['/Data1']  # Data folders, can contain the same as training or new data for consistency.

FPS = 60  # Frame-rate of your video,

# Output directory to where you want the analysis to be stored, including csv, model and plots.
OUTPUT_PATH = '/Users/ahsu/Desktop/bsoid_umap_beta'
MODEL_NAME = 'c57bl6_n3_60min'  # Machine learning model name

# IF YOU'D LIKE TO SKIP PLOTS/VIDEOS, change below PLOT/VID settings to False
PLOT = True
VID = True  # if this is true, make sure direct to the video below AND that you created the two specified folders!

# Create a folder to store extracted images, MAKE SURE THIS FOLDER EXISTS.
FRAME_DIR = '/Users/ahsu/B-SOID/datasets/Data1/0_30min_10fpsPNGs'
# Create a folder to store created video snippets/group, MAKE SURE THIS FOLDER EXISTS.
SHORTVID_DIR = '/Users/ahsu/B-SOID/datasets/Data1/examples'
# Now, pick an example video that corresponds to one of the csv files from the PREDICT_FOLDERS
VID_NAME = '/Users/ahsu/B-SOID/datasets/Data1/2019-04-19_09-34-36cut0_30min.mp4'
ID = 0  # What number would the video be in terms of prediction order? (0=file 1/folder1, 1=file2/folder 1, etc.)

# for semisupervised portion
# CSV_PATH =
