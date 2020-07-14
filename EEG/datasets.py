import os
import pandas as pd
import pickle as pkl
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from EEG.feature_utils import feature_names_len_from_subset, estimate_med_dist, check_if_normalized
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
import torch


class SHHSDataset(Dataset):

    def __init__(self, mode, num_patients, task, normalize_signals, oversample, dataset_dir='filtered', conv_d=1,
                 specific_signal=None, features_subset=None, filter_stage=None, one_slice=False, random_flag=False,
                 num_ch=2, low_sp=False):
        if features_subset is None:
            features_subset = []

        if 'nofilter' in dataset_dir:
            fs = 125
        elif 'filtered' in dataset_dir:
            fs = 80
        else:
            fs = 80
        data_dir = os.path.join(os.getcwd(), 'shhs', 'preprocessed', 'shhs1', dataset_dir)
        if specific_signal is not None:
            s = specific_signal.split('_')
            sig_names, slice = [s[0]], int(s[1])
        else:
            sig_names = pkl.load(open(os.path.join(os.getcwd(), 'shhs', 'results', dataset_dir, f'names_{mode}.pkl'), 'rb'))

        sig_names = sig_names[:int(num_patients)]
        self.waveforms, self.eog, self.emg, self.labels, self.features = {}, {}, {}, {}, {}
        self.load_features(features_subset, random_flag, low_sp, num_ch)
        self.train_std = 23.24
        self.normalize_signals = normalize_signals

        self.sscaler = StandardScaler()
        for sig_name in tqdm(sig_names):
            if dataset_dir == 'eog0.05':
                # sigs, eogs, labels = pkl.load(open(os.path.join(data_dir, f'shhs1-{sig_name}.p'), 'rb'))
                raise NotImplementedError
            elif dataset_dir == 'ALL0.05':
                sigs, eogs, emgs, labels = pkl.load(open(os.path.join(data_dir, f'shhs1-{sig_name}.p'), 'rb'))

            else:
                sigs, labels = pkl.load(open(os.path.join(data_dir, f'shhs1-{sig_name}.p'), 'rb'))

            assert sigs.shape[0] == labels.shape[0]
            num_slices = sigs.shape[0]
            if normalize_signals:
                sigs = sigs / self.train_std

            if specific_signal is not None:
                sig = sigs[slice, :, :].astype('float32')
                if dataset_dir != 'eog0.05':
                    sig = sig[:, fs*60:-30*fs] if one_slice else sig

                if num_ch == 1:
                    sig = np.reshape(sig[0, :], (1, -1))

                self.waveforms[f'{sig_name}_{slice}'] = sig
                self.labels[f'{sig_name}_{slice}'] = labels[slice]

            else:
                for i in range(num_slices):

                    sig = sigs[i, :, :].astype('float32')

                    if dataset_dir == 'eog0.05':
                        eog = eogs[i, :, :].astype('float32')
                        self.eog[f'{sig_name}_{i}'] = eog

                    elif dataset_dir == 'ALL0.05':
                        eog = eogs[i, :, :].astype('float32')
                        emg = emgs[i, :, :].astype('float32')

                        self.eog[f'{sig_name}_{i}'] = eog
                        self.emg[f'{sig_name}_{i}'] = emg

                    else:
                        sig = sig[:, fs*60:-30*fs] if one_slice else sig

                    if num_ch == 1:
                        sig = np.reshape(sig[0, :], (1, -1))

                    # sig = sscaler.fit_transform(sig)
                    self.waveforms[f'{sig_name}_{i}'] = sig
                    self.labels[f'{sig_name}_{i}'] = labels[i]

                if conv_d == 2:
                    sig = sig.reshape(1, sig.shape[0], sig.shape[1])

        self.labels = pd.DataFrame.from_dict(self.labels, orient='index')
        self.labels = self.labels.reset_index()
        self.labels = self.labels.rename(columns={'index': 'signal_name', 0: 'target'})

        # if mode == 'train':
        #     self.sscaler.fit_transform()

        if 'wake_rem' == task:      # wake vs rem -
            self.labels = self.labels.loc[(self.labels.target == 0) | (self.labels.target == 4)]
            self.labels.loc[self.labels.target == 4, 'target'] = 1
        elif 'rem_nrem' == task:        # rem vs nrem check emg or eog
            # self.labels = self.labels.loc[self.labels.target != 0] # todo: change back
            self.labels.loc[self.labels.target != 4, 'target'] = 0
            self.labels.loc[self.labels.target == 4, 'target'] = 1

        if filter_stage is not None:
            self.labels = self.labels[self.labels.target == filter_stage]
            self.labels = self.labels.reset_index()
        self.dataset_size = len(self.labels)
        self.sampler = sampler_func(oversample, self.labels)

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

    def mini_dataset(self):

        data_list = []
        for i in range(5):
            data_list.append(self.labels[self.labels.target == i].sample(n=1000, random_state=42))
        df = pd.concat(data_list)
        df.reset_index()
        self.labels = df
        self.dataset_size = len(self.labels)

    def load_features(self, features_subset, random_flag, low_sp, num_ch):
        # assert (low_sp or 'sw' not in features_subset)  # want low sp and sw? need to merge

        self.feature_len, feature_names = feature_names_len_from_subset(features_subset)
        feature_file_name = get_feature_file_name(low_sp=low_sp, num_ch=num_ch)
        self.features = pkl.load(open(os.path.join(os.getcwd(), 'shhs', 'preprocessed', 'features',
                                                    feature_file_name), 'rb'))
        self.features.set_index('signal_name', inplace=True)
        self.features = self.features[feature_names]
        if random_flag:
            self.features = np.random.rand(self.features.shape[0], self.features.shape[1])
        if self.feature_len > 0:
            self.med_dist = estimate_med_dist(self.features, percentile=50)
        else:
            self.med_dist = 0
        check_if_normalized(self.features)
        return self

    # def sampler_func(self, resample):
    #     total = len(self.labels)
    #     self.labels['weights'] = 1
    #
    #     counts = self.labels.target.value_counts()
    #     sampler = None
    #     if resample:
    #         for i in range(2):
    #             total2 = counts[0] + counts[1]
    #             self.labels.loc[self.labels.target == i, 'weights'] = total2 / counts[i]
    #
    #         sampler = WeightedRandomSampler(weights=torch.DoubleTensor(self.labels.weights), replacement=True,
    #                                         num_samples=total)
    #     return sampler

    def unnormalize_signal(self, signal):
        if self.normalize_signals:
            signal = signal * self.train_std
        return signal


def sampler_func(oversample, labels):
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
    sampler = loader.dataset.sampler_func(resample=new_state)
    loader = DataLoader(dataset=loader.dataset, batch_size=loader.batch_size, shuffle=False,
                        sampler=sampler)
    return loader


def init_datasets(task, oversample, normalize_signals, num_patients=200, dataset_dir='filtered0.05', batch_size=32,
                  conv_d=1, features_subset=None, one_slice=False, modes=None, random_flag=False, num_ch=2,
                  low_sp=False, filter_stage=None):
    if features_subset is None:
        features_subset = []
    if modes is None:
        modes = ['train', 'val', 'test']
    for mode in modes:
        assert mode in ['train', 'val', 'test']  # Check your spelling!
    loaders = []
    if 'train' in modes:
        train_dataset = SHHSDataset(mode="train", num_patients=num_patients, dataset_dir=dataset_dir, conv_d=conv_d,
                                    features_subset=features_subset, one_slice=one_slice, random_flag=random_flag,
                                    num_ch=num_ch, low_sp=low_sp, filter_stage=filter_stage, task=task,
                                    oversample=oversample, normalize_signals=normalize_signals)
        shuffle = True
        if train_dataset.sampler is not None:
            shuffle = False
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, sampler=train_dataset.sampler)
        loaders += [train_loader]

    if 'val' in modes:
        val_dataset = SHHSDataset(mode="valid", num_patients=num_patients, dataset_dir=dataset_dir, conv_d=conv_d,
                                  features_subset=features_subset, one_slice=one_slice, random_flag=random_flag,
                                  num_ch=num_ch, low_sp=low_sp, filter_stage=filter_stage, task=task,
                                  oversample=oversample, normalize_signals=normalize_signals)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, sampler=val_dataset.sampler)
        loaders += [val_loader]

    if 'test' in modes:
        test_dataset = SHHSDataset(mode="test", num_patients=num_patients, dataset_dir=dataset_dir, conv_d=conv_d,
                                   features_subset=features_subset, one_slice=one_slice, random_flag=random_flag,
                                   num_ch=num_ch, low_sp=low_sp, filter_stage=filter_stage, task=task,
                                   oversample=oversample, normalize_signals=normalize_signals)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, sampler=test_dataset.sampler)
        loaders += [test_loader]

    assert len(loaders) > 0
    if len(loaders) == 1:
        loaders = loaders[0]
    return loaders


def init_specific_signal(specific_signal_name, dataset_dir='filtered0.05', conv_d=1, features_to_compute=None,
                         low_sp=False, num_ch=2, one_slice=False):
    if features_to_compute is None:
        features_to_compute = []
    test_dataset = SHHSDataset("test", num_patients=1, dataset_dir=dataset_dir, conv_d=conv_d, num_ch=num_ch,
                               specific_signal=specific_signal_name, features_subset=features_to_compute, low_sp=low_sp, one_slice=one_slice)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    return test_loader


def get_feature_file_name(low_sp, num_ch):
    if num_ch == 1:
        if low_sp:
            feature_file_name = 'splow_freq_sig_1ch.pkl'
        else:
            feature_file_name = 'sp_freq_sw_1ch.pkl'    # 'sp_freq_sig_1ch.pkl'
    else:
        if low_sp:
            feature_file_name = 'splow_freq_sig.pkl'
        else:
            # This one for sw, sp, rem:
            feature_file_name = 'sp_sw_rem.pkl'
            # This one for freq:
            # feature_file_name = 'frequency_relative.pkl' #'frequency_relative_all.pkl'  #'frequency_relative.pkl'    #'frequency_nonorm.pkl'  # 'freq_sw.pkl'    # 'sp_freq_sig_norm_norm.pkl'
    return feature_file_name


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
        fs = 80  # we extracted 5-fold CV splits only for the downsampled data
        data_dir = os.path.join(os.getcwd(), 'shhs', 'preprocessed', 'shhs1', 'ALL0.05', 'held_out_data')
        dataset_path = os.path.join(data_dir, f'{mode}_{idx}.pkl')

        self.labels = pd.read_pickle(dataset_path)
        self.dataset_size = len(self.labels)

        self.sampler = sampler_func(oversample, self.labels)

        return

    def __getitem__(self, index):
        item = self.labels.iloc[index]
        target = item['target']
        signal_name = self.labels.iloc[index]['signal_name']

        return signal_name, target

    def __len__(self):
        return self.dataset_size

