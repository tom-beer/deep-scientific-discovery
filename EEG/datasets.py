import os
import pandas as pd
import pickle as pkl
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from EEG.feature_utils import feature_names_len_from_subset, estimate_med_dist, check_if_normalized


class SHHSDataset(Dataset):

    def __init__(self, mode, task, normalize_signals, balanced_dataset, specific_signal=None, features_subset=None,
                 filter_stage=None, idxs=None):
        if features_subset is None:
            features_subset = []

        fs = 80
        eeg_proj_dir = os.path.dirname(os.path.realpath(__file__))
        data_dir = os.path.join(eeg_proj_dir, 'shhs', 'preprocessed', 'shhs1', 'ALL0.05')
        if specific_signal is not None:
            s = specific_signal.split('_')
            sig_names, slice = [s[0]], int(s[1])
        else:
            sig_names = pkl.load(open(os.path.join(eeg_proj_dir, 'shhs', 'results', 'ALL0.05', f'names_{mode}.pkl'), 'rb'))

        self.waveforms, self.eog, self.emg, self.labels, self.features = {}, {}, {}, {}, {}
        self.load_features(features_subset)
        self.train_std = 23.24
        self.normalize_signals = normalize_signals

        for sig_name in sig_names:
            sigs, eogs, emgs, labels = pkl.load(open(os.path.join(data_dir, f'shhs1-{sig_name}.p'), 'rb'))

            assert sigs.shape[0] == labels.shape[0]
            num_slices = sigs.shape[0]
            if normalize_signals:
                sigs = sigs / self.train_std

            if specific_signal is not None:
                sig = sigs[slice, :, :].astype('float32')
                sig = sig[:, fs*60:-30*fs]

                self.waveforms[f'{sig_name}_{slice}'] = sig
                self.labels[f'{sig_name}_{slice}'] = labels[slice]

            else:
                for i in range(num_slices):

                    sig = sigs[i, :, :].astype('float32')
                    eog = eogs[i, :, :].astype('float32')
                    emg = emgs[i, :, :].astype('float32')

                    self.eog[f'{sig_name}_{i}'] = eog
                    self.emg[f'{sig_name}_{i}'] = emg

                    self.waveforms[f'{sig_name}_{i}'] = sig
                    self.labels[f'{sig_name}_{i}'] = labels[i]

        self.labels = pd.DataFrame.from_dict(self.labels, orient='index')
        self.labels = self.labels.reset_index()
        self.labels = self.labels.rename(columns={'index': 'signal_name', 0: 'target'})
        if idxs is not None:
            self.labels = self.labels.iloc[idxs]

        if task == 'wake_rem':
            self.labels = self.labels.loc[(self.labels.target == 0) | (self.labels.target == 4)]
            self.labels.loc[self.labels.target == 4, 'target'] = 1
        elif task == 'rem_nrem':
            self.labels.loc[self.labels.target != 4, 'target'] = 0
            self.labels.loc[self.labels.target == 4, 'target'] = 1

        if filter_stage is not None:
            self.labels = self.labels[self.labels.target == filter_stage]
            self.labels = self.labels.reset_index()
        self.dataset_size = len(self.labels)
        self.sampler = balance_classes(balanced_dataset, self.labels)

    def __getitem__(self, index, random_flag=False):
        item = self.labels.iloc[index]
        target = item['target']
        signal_name = item['signal_name']
        signal = self.waveforms[signal_name]
        features = self.features.loc[signal_name].values if self.feature_len > 0 else np.zeros([1])
        return signal, target, signal_name, features.astype('float32')

    def get_eog(self, signal_name):
        return self.eog[signal_name[0]]

    def get_emg(self, signal_name):
        return self.emg[signal_name[0]]

    def __len__(self):
        return self.dataset_size

    def load_features(self, features_subset):

        self.feature_len, feature_names = feature_names_len_from_subset(features_subset)
        feature_file_name = 'frequency_relative.pkl'
        self.features = pkl.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                   'shhs', 'preprocessed', 'features', feature_file_name), 'rb'))
        self.features = self.features[feature_names]
        if self.feature_len > 0:
            self.med_dist = estimate_med_dist(self.features, percentile=50)
        else:
            self.med_dist = 0
        check_if_normalized(self.features)
        return self

    def unnormalize_signal(self, signal):
        if self.normalize_signals:
            signal = signal * self.train_std
        return signal


def balance_classes(oversample, labels):
    total = len(labels)
    labels['weights'] = 1

    counts = labels.target.value_counts()
    sampler = None
    if oversample:
        for i in range(2):
            total2 = counts[0] + counts[1]
            labels.loc[labels.target == i, 'weights'] = total2 / counts[i]

        sampler = WeightedRandomSampler(weights=torch.DoubleTensor(labels.weights.values), replacement=True,
                                        num_samples=total)
    return sampler


def change_sampler(loader, new_state=False):
    sampler = loader.dataset.balance_classes(resample=new_state)
    loader = DataLoader(dataset=loader.dataset, batch_size=loader.batch_size, shuffle=False,
                        sampler=sampler)
    return loader


def init_datasets(task, balanced_dataset, normalize_signals=True, batch_size=32, features_subset=None, modes=None, filter_stage=None):
    if features_subset is None:
        features_subset = []
    if modes is None:
        modes = ['train', 'val', 'test']
    for mode in modes:
        assert mode in ['train', 'val', 'test']
    loaders = []
    if 'train' in modes:
        train_dataset = SHHSDataset(mode="train", features_subset=features_subset, filter_stage=filter_stage, task=task,
                                    balanced_dataset=balanced_dataset, normalize_signals=normalize_signals)
        shuffle = True
        if train_dataset.sampler is not None:
            shuffle = False
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, sampler=train_dataset.sampler)
        loaders += [train_loader]

    if 'val' in modes:
        val_dataset = SHHSDataset(mode="valid", features_subset=features_subset, filter_stage=filter_stage, task=task,
                                  balanced_dataset=balanced_dataset, normalize_signals=normalize_signals)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, sampler=val_dataset.sampler)
        loaders += [val_loader]

    if 'test' in modes:
        test_dataset = SHHSDataset(mode="test", features_subset=features_subset, filter_stage=filter_stage, task=task,
                                   balanced_dataset=balanced_dataset, normalize_signals=normalize_signals)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, sampler=test_dataset.sampler)
        loaders += [test_loader]

    assert len(loaders) > 0
    if len(loaders) == 1:
        loaders = loaders[0]
    return loaders


def init_gap_datasets(modes, idx, oversample, batch_size):

    if modes is None:
        modes = ['train', 'val', 'test']
    for mode in modes:
        assert mode in ['train', 'val', 'test']  # Check your spelling!
    loaders = []
    if 'train' in modes:
        train_dataset = GapDataset(mode="train", idx=idx, oversample=oversample)
        shuffle = True
        if train_dataset.sampler is not None:
            shuffle = False
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, sampler=train_dataset.sampler)
        loaders += [train_loader]

    if 'val' in modes:
        val_dataset = GapDataset(mode="val", idx=idx, oversample=oversample)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, sampler=val_dataset.sampler)
        loaders += [val_loader]

    if 'test' in modes:
        test_dataset = GapDataset(mode="test", idx=idx, oversample=oversample)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, sampler=test_dataset.sampler)
        loaders += [test_loader]

    assert len(loaders) > 0
    if len(loaders) == 1:
        loaders = loaders[0]
    return loaders


class GapDataset(Dataset):
    def __init__(self, mode, idx, oversample):
        data_dir = os.path.join(os.getcwd(), 'shhs', 'preprocessed', 'shhs1', 'ALL0.05', 'held_out_data')
        dataset_path = os.path.join(data_dir, f'{mode}_{idx}.pkl')

        self.labels = pd.read_pickle(dataset_path)
        self.dataset_size = len(self.labels)

        self.sampler = balance_classes(oversample, self.labels)

        return

    def __getitem__(self, index):
        item = self.labels.iloc[index]
        target = item['target']
        signal_name = self.labels.iloc[index]['signal_name']

        return signal_name, target

    def __len__(self):
        return self.dataset_size


def create_kfoldcv_loaders(task, balanced_dataset, normalize_signals=True, batch_size=32, features_subset=None):
    num_folds = 5
    kfoldcv_testloaders = []
    test_dataset = SHHSDataset("test", task=task, normalize_signals=normalize_signals,
                               balanced_dataset=balanced_dataset, features_subset=features_subset)

    np.random.seed(24)
    idxs = (np.random.multinomial(1, 0.2 * np.ones(5).ravel(), size=len(test_dataset)) == 1).argmax(1).astype(int)

    for i_fold in range(num_folds):
        idx_test = idxs == i_fold
        idx_val = idxs == (i_fold + 1) % 5
        idx_train = ((idxs == (i_fold + 2) % 5) |
                     (idxs == (i_fold + 3) % 5) |
                     (idxs == (i_fold + 4) % 5))

        train_dataset = SHHSDataset("test", task=task, normalize_signals=normalize_signals,
                                    balanced_dataset=balanced_dataset, features_subset=features_subset, idxs=idx_train)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False,
                                                   sampler=train_dataset.sampler)
        val_dataset = SHHSDataset("test", task=task, normalize_signals=normalize_signals,
                                  balanced_dataset=balanced_dataset, features_subset=features_subset, idxs=idx_val)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,
                                                 sampler=val_dataset.sampler)
        test_dataset = SHHSDataset("test", task=task, normalize_signals=normalize_signals,
                                   balanced_dataset=balanced_dataset, features_subset=features_subset, idxs=idx_test)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                                  sampler=test_dataset.sampler)
        kfoldcv_testloaders.append((train_loader, val_loader, test_loader))
    return kfoldcv_testloaders
