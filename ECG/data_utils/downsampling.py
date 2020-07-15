import os
import json
import pandas as pd
import numpy as np
from scipy.signal import resample


def downsample(mode, data_dir, new_freq, old_freq=300, old_len=18000):
    dataset_path = os.path.join(data_dir, 'training', mode)
    json_dict = json.load(open(os.path.join(dataset_path, 'labels', 'labels.json')))
    labels = pd.DataFrame.from_dict(data=json_dict, orient='index')
    labels = labels.reset_index()
    labels = labels.rename(columns={'index': 'signal', 0: 'target'})
    waveform_path = os.path.join(dataset_path, 'waveforms')
    dataset_size = len(labels)
    ds_rate = int(old_len // (old_freq/new_freq) + 1)  # 5400

    for idx in range(dataset_size):
        signal_name = labels.iloc[idx]['signal']
        file_name = os.path.join(waveform_path, signal_name) + '.npy'
        signal = np.load(file_name)
        signal = resample(signal, ds_rate)
        np.save(file_name, signal)
