import json
import os
import numpy as np
import pandas as pd
import pickle as pkl
import wfdb
import torch
from wfdb import processing
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import WeightedRandomSampler

import ECG.feature_utils as futil


def single_rr(sig):
    xqrs = wfdb.processing.XQRS(sig=sig.ravel(), fs=300)
    xqrs.detect()
    rr = wfdb.processing.calc_rr(xqrs.qrs_inds, fs=300, min_rr=xqrs.rr_min, max_rr=xqrs.rr_max)
    return rr


class ECGDataset(Dataset):

    def __init__(self, mode, feature_subset='all', feature_opt=None, oversample=False, is_baseline=False, naf=False,
                 idxs=None):

        self.feature_subset = feature_subset
        self.feature_opt = feature_opt
        self.feature_subset = feature_subset
        self.is_baseline = is_baseline
        self.data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'training')
        self.dataset_path = os.path.join(self.data_dir, mode)
        self.idxs = idxs

        # These options are called only from plot_utils.py, so need to go 1 folder up
        # if (oversample == 'af') or (oversample == 'normal'):
            # assert 0  # need to check if this still works..
            # main_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
            # dataset_path = os.path.join(main_dir, self.dataset_path)
            # data_dir = os.path.join(main_dir, data_dir)

        self.load_labels()

        if naf:
            # unify normal and other classes to label=0, AF is label=1
            self.labels.loc[self.labels.target == 2, 'target'] = 0

        if oversample == 'af':
            self.labels = self.labels[self.labels.target == 1]
            oversample = 'none'
        elif oversample == 'normal':
            self.labels = self.labels[self.labels.target == 0]
            oversample = 'none'

        self.labels = self.labels.reset_index()
        self.labels.drop(columns=['index'], inplace=True)

        self.waveform_path = os.path.join(self.dataset_path, 'waveforms')

        self.load_waveforms()

        self.feature_len = 0
        if oversample is not 'none':
            self.sampler = self.balance_classes()

        self.load_features()

    def __getitem__(self, index):
        item = self.labels.iloc[index]
        target = item['target']
        signal = self.waveforms[index, :]
        signal = signal.reshape(1, signal.shape[0])
        feature = np.zeros(1, dtype='float32')
        feature_rep = np.zeros(1, dtype='float32')

        signal_name = self.labels.iloc[index]['signal']
        real_feature = self.real_features.loc[signal_name.split(sep='_')[0]].values

        if self.feature_len > 0:
            feature = self.features.loc[signal_name.split(sep='_')[0]].values.astype('float32')
            feature_rep = feature  # relic from a time we used representations learned from another network

        return signal.astype('float32'), target, feature, real_feature.astype('float32'), signal_name, feature_rep

    def __len__(self):
        return self.dataset_size

    def load_labels(self):
        json_dict = json.load(open(os.path.join(self.dataset_path, 'labels', 'labels.json')))
        self.labels = pd.DataFrame.from_dict(data=json_dict, orient='index')
        self.labels = self.labels.reset_index()
        self.labels = self.labels.rename(columns={'index': 'signal', 0: 'target'})
        # exclude noisy
        self.labels = self.labels[self.labels.target != 3]
        self.labels = self.labels.reset_index()
        self.labels.drop(columns=['index'], inplace=True)

        if self.idxs is not None:
            self.labels = self.labels.iloc[self.idxs]

    def load_waveforms(self):
        self.dataset_size = len(self.labels)
        self.waveforms = np.ones((self.dataset_size, 5400))
        for idx in range(self.dataset_size):
            signal_name = self.labels.iloc[idx]['signal']
            self.waveforms[idx, :] = np.load(os.path.join(self.waveform_path, signal_name) + '.npy')

    def load_features(self):
        df = pd.read_csv(os.path.join(self.data_dir, 'features_normalized.csv'), index_col=[0]).fillna(-1)
        assert (len(list(df))) == 17
        self.real_features = df

        if ('HSIC' in self.feature_opt.lower()) or ('concat' in self.feature_opt.lower()):

            if 'rr' in self.feature_subset:
                feature_names = futil.rr_feature_names
            elif 'all' in self.feature_subset:
                feature_names = futil.all_feature_names
            elif 'p_wave' in self.feature_subset:
                feature_names = futil.p_wave_feature_names

            self.features = df[feature_names]
            self.feature_len = len(list(self.features))

            if self.is_baseline:
                self.feature_len = 0

    def balance_classes(self):
        total = len(self.labels)
        self.labels['weights'] = 1

        counts = self.labels.target.value_counts()
        for i in range(2):
            total2 = counts[0] + counts[1]
            self.labels.loc[self.labels.target == i, 'weights'] = total2 / counts[i]
        self.labels.loc[self.labels.target == 2, 'weights'] = 0.0001  # just a small probability
        sampler = WeightedRandomSampler(weights=torch.DoubleTensor(self.labels.weights), replacement=True,
                                        num_samples=total)
        return sampler

    def generate_all_rrs(self):
        max_len = 0
        rr_dict = {}
        for idx in range(self.dataset_size):
            signal_name = self.labels.iloc[idx]['signal']
            rr_dict[signal_name] = single_rr(self.waveforms[idx, :])
            if max_len < rr_dict[signal_name].shape[0]:
                max_len = rr_dict[signal_name].shape[0]
        for k, v in rr_dict.items():
            len_padding = max_len - v.shape[0]
            padding = np.ones(len_padding) * 0
            new_rr = np.append(v, padding)
            rr_dict[k] = new_rr

        rr_df = pd.DataFrame.from_dict(rr_dict, orient='index')
        rr_df.to_csv(os.path.join(os.getcwd(), 'rr.csv'))
        return


class GapDataset(Dataset):
    def __init__(self, mode, idx, data_dir=os.path.join(os.getcwd(), 'data', 'held_out_data'), oversample='None',
                 less_normal=False, file_name=None, naf=False):
        dataset_path = os.path.join(data_dir, f'{mode}_{idx}.pkl')
        self.labels = pd.read_pickle(dataset_path)

        if less_normal:
            with open(os.path.join(file_name, f'{file_name}_{mode}_dropped.pkl'), 'wb') as handle:
                drop_indices = pkl.load(handle, protocol=pkl.HIGHEST_PROTOCOL)
            self.labels = self.labels.drop(drop_indices)
        if naf:
            self.labels.loc[self.labels.target == 2, 'target'] = 0
        self.labels = self.labels[self.labels.target != 3]

        self.labels = self.labels.reset_index()
        self.labels.drop(columns={'index'}, inplace=True)
        self.dataset_size = len(self.labels)

        self.sampler = None

        self.sampler = self.sampler_func(oversample)

        return

    def __getitem__(self, index):
        item = self.labels.iloc[index]
        target = item['target']
        signal_name = self.labels.iloc[index]['signal']

        return signal_name, target

    def __len__(self):
        return self.dataset_size

    def sampler_func(self, odds_type):
        print(f'{odds_type}')
        total = len(self.labels)
        self.labels['weights'] = 1

        counts = self.labels.target.value_counts()
        sampler = None
        if odds_type.lower() != 'none':
            if odds_type == '131':
                self.labels.loc[self.labels.target == 1, 'weights'] = 3
            elif odds_type == 'balanced':
                counts = self.labels.target.value_counts()
                for i in range(3):
                    self.labels.loc[self.labels.target == i, 'weights'] = total / counts[i]
            elif odds_type == '25':
                ratios = np.array([1, 2, 1])
                for i in range(3):
                    self.labels.loc[self.labels.target == i, 'weights'] = total / counts[i] * ratios[i]
            elif odds_type == '50':
                for i in range(2):
                    total2 = counts[0] + counts[1]
                    self.labels.loc[self.labels.target == i, 'weights'] = total2 / counts[i]
                self.labels.loc[self.labels.target == 2, 'weights'] = 0.0008
            sampler = WeightedRandomSampler(weights=torch.DoubleTensor(self.labels.weights), replacement=True,
                                            num_samples=total)

        return sampler


def create_dataloaders(batch_size, feature_subset, feature_opt, naf):
    train_dataset = ECGDataset("train", feature_subset=feature_subset, feature_opt=feature_opt,
                               oversample=False, naf=naf)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False,
                                               sampler=train_dataset.sampler)

    val_dataset = ECGDataset("val", feature_subset=feature_subset, feature_opt=feature_opt,
                             oversample=False, naf=naf)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,
                                             sampler=val_dataset.sampler)

    test_dataset = ECGDataset("test", feature_subset=feature_subset, feature_opt=feature_opt,
                              oversample=False, naf=naf)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                              sampler=test_dataset.sampler)

    return train_loader, val_loader, test_loader


def create_kfoldcv_loaders(batch_size, feature_subset, feature_opt, naf):
    num_folds = 5
    kfoldcv_testloaders = []
    test_dataset = ECGDataset("test", feature_subset=feature_subset, feature_opt=feature_opt,
                              oversample=False, naf=naf)

    np.random.seed(24)
    idxs = (np.random.multinomial(1, 0.2 * np.ones(5).ravel(), size=len(test_dataset)) == 1).argmax(1).astype(int)

    for i_fold in range(num_folds):
        idx_test = idxs == i_fold
        idx_val = idxs == (i_fold + 1) % 5
        idx_train = ((idxs == (i_fold + 2) % 5) |
                     (idxs == (i_fold + 3) % 5) |
                     (idxs == (i_fold + 4) % 5))

        train_dataset = ECGDataset("test", feature_subset=feature_subset, feature_opt=feature_opt,
                                   oversample=False, naf=naf, idxs=idx_train)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False,
                                                   sampler=train_dataset.sampler)
        val_dataset = ECGDataset("test", feature_subset=feature_subset, feature_opt=feature_opt,
                                 oversample=False, naf=naf, idxs=idx_val)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,
                                                 sampler=val_dataset.sampler)
        test_dataset = ECGDataset("test", feature_subset=feature_subset, feature_opt=feature_opt,
                                  oversample=False, naf=naf, idxs=idx_test)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                                  sampler=test_dataset.sampler)
        kfoldcv_testloaders.append((train_loader, val_loader, test_loader))
    return kfoldcv_testloaders
