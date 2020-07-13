import torch.utils.data
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from yasa import rem_detect

from EEG.features import compute_features, feature_names_len_from_subset, normalize_features
from EEG.datasets import DSMDataset


def extract_features(features_to_compute, save_name, low, modes=None, mini=False, normalize=False, num_chan=2):

    if modes is None:
        modes = ['train', 'valid', 'test']
    for mode in modes:
        assert mode in ['train', 'valid', 'test']  # Check your spelling!

    channel_names = ['EEG', 'EEG(sec)'][:num_chan]

    dataset_dir = 'ALL0.05'
    fs = 80 if ds else 125

    features = {}
    for mode in modes:
        dataset = DSMDataset(mode, num_patients=1e8, dataset_dir=dataset_dir, conv_d=1, features_subset=[],
                             one_slice=True, num_ch=num_chan, task='rem_nrem', normalize_signals=False,
                             oversample=False)

        if mini:
            dataset.mini_dataset()
        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)

        for batch_idx, (signal, target, signal_name, _) in enumerate(tqdm(loader)):
            signal = signal.numpy().squeeze()
            signal = signal[:num_chan, :]
            features_curr = compute_features(signal, low=low, fs=fs, ch_names=channel_names,
                                                        features_to_compute=features_to_compute)
            if 'rem' in features_to_compute:
                eog = dataset.get_eog(signal_name)
                rem = rem_detect(loc=eog[1, :], roc=eog[0, :], sf=50)
                if rem is None:
                    rem_feature = 0.
                else:
                    rem_feature = 1.
                features_curr = np.hstack([features_curr, rem_feature])
            features[signal_name[0]] = features_curr


    features_df = pd.DataFrame.from_dict(data=features, columns=feature_names_len_from_subset(features_to_compute)[1],
                                         orient='index')
    if normalize:
        features_df = normalize_features(features_df)

    # features_df = features_df.reset_index()
    # features_df = features_df.rename(columns={'index': 'signal_name'})
    features_df.fillna(0, inplace=True)
    pd.to_pickle(features_df, os.path.join(os.getcwd(), 'shhs', 'preprocessed', 'features', save_name))

    return features_df


features_to_compute = ['spindle', 'sw', 'rem']
normalize = True
ds = True
num_chan = 2
low = False
save_name = 'sp_sw_rem.pkl'
features_df = extract_features(features_to_compute, modes=['train', 'valid', 'test'], mini=False, normalize=normalize,
                               num_chan=num_chan, save_name=save_name, low=low)

# Check that we can read this file:
# df = pd.read_pickle(os.path.join(os.getcwd(), 'shhs', 'preprocessed', 'features','features_0.05_freq_spindle.pkl'))
