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

    def __init__(self, mode, feature_subset='all', feature_opt=None, oversample=False, is_baseline=False,
                 less_normal=False, file_name=None, with_cam=False, naf=False):
        self.feature_subset = feature_subset
        data_dir = os.path.join('data', 'training')

        dataset_path = os.path.join(data_dir, mode)

        # These options are called only from plot_utils.py, so need to go 1 folder up
        if (oversample == 'af') or (oversample == 'normal'):
            main_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
            dataset_path = os.path.join(main_dir, dataset_path)
            data_dir = os.path.join(main_dir, data_dir)

        json_dict = json.load(open(os.path.join(dataset_path, 'labels', 'labels.json')))
        self.labels = pd.DataFrame.from_dict(data=json_dict, orient='index')
        self.labels = self.labels.reset_index()
        self.labels = self.labels.rename(columns={'index': 'signal', 0: 'target'})
        # excluding noisy
        if less_normal:
            self.change_labels(file_name, mode)
        if naf:
            self.labels.loc[self.labels.target == 2, 'target'] = 0

        if with_cam:
            self.get_cams(file_name, mode)
        self.labels = self.labels[self.labels.target != 3]

        if oversample == 'af':
            self.labels = self.labels[self.labels.target == 1]
            oversample = 'none'
        elif oversample == 'normal':
            self.labels = self.labels[self.labels.target == 0]
            oversample = 'none'

        self.labels = self.labels.reset_index()
        self.labels.drop(columns=['index'], inplace=True)

        self.waveform_path = os.path.join(dataset_path, 'waveforms')
        self.dataset_size = len(self.labels)

        self.waveforms = np.ones((self.dataset_size, 5400))
        for idx in range(self.dataset_size):
            signal_name = self.labels.iloc[idx]['signal']
            signal = np.load(os.path.join(self.waveform_path, signal_name) + '.npy')
            self.waveforms[idx, :] = signal
        self.feature_len = 0
        if oversample != 'none':
            self.sampler = self.balance_classes()

        feature_len = 0
        df = pd.read_csv(os.path.join(data_dir, 'features_normalized.csv'), index_col=[0]).fillna(-1)
        assert (len(list(df))) == 17
        self.real_features = df

        if ('HSIC' in feature_opt) or ('concat' in feature_opt.lower()):
            if 'rr' in feature_subset:
                self.features = df[futil.rr_feature_names]
                extracted_rep_file_name = 'rr_ext_rep_features.pkl'
                feature_len = len(list(self.features))
            elif 'all' in feature_subset:
                self.features = df
                extracted_rep_file_name = 'all_ext_rep_features_new.pkl'
                feature_len = len(list(self.features))
            elif 'p_wave' in feature_subset:
                self.features = df[futil.p_wave_feature_names]
                extracted_rep_file_name = 'p_wave_ext_rep_features_new.pkl'
                feature_len = len(list(self.features))
            elif feature_subset == 'full_waveform':
                self.features = df[futil.full_waveform_feature_names]
                feature_len = len(list(self.features))
            elif feature_subset == 'random':
                self.features = pd.DataFrame(np.random.normal(size=df.values.shape), index=df.index)
                feature_len = len(list(self.features))
            elif feature_subset == 'rr_local':
                main_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
                df = pd.read_csv(os.path.join(main_dir, 'data', 'rr_local', 'rr_locals_90_test_filtered7.csv'))
                df.set_index('Unnamed: 0', inplace=True)
                self.features = df
                feature_len = len(list(self.features))
            self.feature_len = feature_len

            if 'fcnet' in feature_subset:
                with open(os.path.join(data_dir, extracted_rep_file_name), 'rb') as handle:
                    rep_features = pkl.load(handle)
                self.features_rep = pd.DataFrame.from_dict(rep_features, orient='index')
                self.feature_len = len(list(self.features_rep))
            if is_baseline:
                self.feature_len = 0

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

            if 'fcnet' in self.feature_subset:
                feature_rep = self.features_rep.loc[signal_name].values.astype('float32')
                # trying concat of original feature after neural representation
                # feature_rep = np.hstack([feature_rep, feature])
                # self.in_size = self.in_size + len(real_feature)

            else:
                feature_rep = feature

        return signal.astype('float32'), target, feature, real_feature.astype('float32'), signal_name, feature_rep

    def __len__(self):
        return self.dataset_size

    def get_cams(self, file_name, mode):
        with open(os.path.join(file_name, f'{file_name}_ucam_{mode}.pkl'), 'wb') as handle:
            cams = pkl.load(handle, protocol=pkl.HIGHEST_PROTOCOL)
        self.cams = cams
        # todo: still need to update carefully in get item and all the following enumerators

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


class FeatureDataset(Dataset):
    def __init__(self, mode, included_subset='rr', excluded_subset='p_wave'):

        assert(included_subset in ['rr', 'p_wave'])  # implement otherwise if necessary
        assert(excluded_subset in ['rr', 'p_wave'])  # implement otherwise if necessary
        data_dir = 'data/training'

        dataset_path = os.path.join(data_dir, f'{mode}')

        self.df = pd.read_csv(os.path.join(data_dir, 'features_normalized.csv'), index_col=[0]).fillna(-1)
        json_dict = json.load(open(os.path.join(dataset_path, 'labels', 'labels.json')))
        self.labels = pd.DataFrame.from_dict(data=json_dict, orient='index')
        self.labels = self.labels.reset_index()
        self.labels = self.labels.rename(columns={'index': 'signal', 0: 'target'})
        self.labels = self.labels[self.labels.target != 3]
        self.labels = self.labels.reset_index()
        self.labels.drop(columns=['index'], inplace=True)

        self.included_names = futil.rr_feature_names if included_subset == 'rr' else futil.p_wave_feature_names
        self.excluded_names = futil.rr_feature_names if excluded_subset == 'rr' else futil.p_wave_feature_names

    def __getitem__(self, index):
        signal_name = self.labels.iloc[index]['signal']
        signal_name = signal_name.split(sep='_')[0]
        inc_features = self.df.loc[signal_name, self.included_names].values
        exc_features = self.df.loc[signal_name, self.excluded_names].values

        return signal_name, inc_features.astype('float32'), exc_features.astype('float32')

    def __len__(self):
        return len(self.labels)


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
