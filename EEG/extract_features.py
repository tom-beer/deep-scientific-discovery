import torch.utils.data
import pandas as pd
import numpy as np
import os
from yasa import rem_detect
from tqdm import tqdm

from EEG.feature_utils import compute_features, feature_names_len_from_subset, normalize_features
from EEG.datasets import SHHSDataset


def extract_features(features_to_compute, save_name, modes=None, normalize=False):

    if modes is None:
        modes = ['train', 'valid', 'test']
    for mode in modes:
        assert mode in ['train', 'valid', 'test']  # Check your spelling!

    channel_names = ['EEG', 'EEG(sec)']

    fs = 80

    features = {}
    for mode in modes:
        dataset = SHHSDataset(mode, features_subset=[], task='rem_nrem', normalize_signals=False, oversample=False)

        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)

        for batch_idx, (signal, target, signal_name, _) in enumerate(tqdm(loader)):
            signal = signal.numpy().squeeze()
            features_curr = compute_features(signal, low=False, fs=fs, ch_names=channel_names,
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
save_name = 'sp_sw_rem.pkl'
features_df = extract_features(features_to_compute, modes=['train', 'valid', 'test'], normalize=normalize,
                               save_name=save_name)
