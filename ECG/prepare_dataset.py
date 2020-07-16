import urllib.request
import os
import warnings
import sys
from ECG.data_utils.physionet2017 import Physionet2017DB
from ECG.data_utils.training_dataset import TrainingDataset
from ECG.data_utils.downsampling import downsample
sys.path.append('..')
warnings.filterwarnings('ignore')


download_url = "https://archive.physionet.org/challenge/2017/training2017.zip"
zip_dest = os.path.join('ECG', 'data', 'dataset', 'raw', 'training2017.zip')
DATA_DIR = os.path.join('ECG', 'data')
dir_name = os.path.dirname(zip_dest)
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

if os.path.exists(zip_dest):
    print("Training data available")
else:
    print("Downloading training data from PhysioNet's website")
    urllib.request.urlretrieve(download_url, zip_dest);
    print('Done')


# Initialize
physionet_2017_db = Physionet2017DB()

# Generate raw database
physionet_2017_db.generate_raw_db()

# Generate processed database
physionet_2017_db.generate_processed_db()

# Inputs
path_labels=os.path.join(DATA_DIR, 'dataset', 'processed', 'labels')
path_save=os.path.join(DATA_DIR, 'training')
classes = ['N', 'A', 'O', '~']
datasets = {'train': 0.7, 'val': 0.15, 'test': 0.15}
duration = 60.
fs = 300

# Get training dataset
train_db = TrainingDataset(path_labels=path_labels, path_save=path_save, duration=duration,
                           classes=classes, datasets=datasets, augment=False, fs=fs)

# Generate dataset
train_db.generate()

# Downasample signals
print('Started downsampling')
for mode in ['train', 'val', 'test']:
    downsample(mode=mode, data_dir=DATA_DIR, new_freq=90, old_freq=300, old_len=18000)
print('Complete!')
