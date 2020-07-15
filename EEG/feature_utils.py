from yasa import bandpower, spindles_detect_multi, sw_detect, spindles_detect, sw_detect_multi
import numpy as np
import pickle as pkl
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import pairwise_distances as pdist
import os


def estimate_med_dist(features, num_slices=5000, percentile=50):

    pdists = pdist(features.values[:num_slices, :], metric='euclidean').reshape(-1, 1)  # to handle sparse data
    med_dist = np.percentile(pdists[pdists > np.finfo(float).eps * 10], percentile)
    med_dist = np.max((0.05, med_dist))
    return med_dist


def check_if_normalized(features):
    normalized = np.all(features.values <= 1) & np.all(features.values >= 0)
    if not normalized:
        print('ATTENTION! You are using unnormalized features!')


def feature_names_len_from_subset(features_subsets):
    possible_feature_subsets = ['spindle', 'num_spindles', 'sw', 'frequency', 'rem']
    for feature_subset in features_subsets:
        assert feature_subset in possible_feature_subsets

    names_list = []
    if ('spindle' in features_subsets) and ('num_spindles' in features_subsets):
        print('got spindle AND num_spindles')
        raise NotImplementedError
    if 'frequency' in features_subsets:
        feature_names = ['Freq_Delta', 'Freq_Theta', 'Freq_Alpha', 'Freq_Beta'] #, '']#]  # 'Freq_Gamma']'Freq_Delta', 'Freq_Theta',
        names_list += feature_names
    if 'sw' in features_subsets:
        feature_names = ['SW_Duration', 'SW_ValNegPeak', 'SW_ValPosPeak', 'SW_PTP', 'SW_Slope', 'SW_Frequency', 'SW_Ch1', 'SW_Ch2']
        names_list += feature_names
    if 'spindle' in features_subsets:
        feature_names = ['SP_Amplitude', 'SP_RMS', 'SP_AbsPower', 'SP_RelPower', 'SP_Oscillations', 'SP_Ch1', 'SP_Ch2']
        names_list += feature_names
    if 'rem' in features_subsets:
        feature_names = ['rem_flag']
        names_list += feature_names
    elif 'num_spindles' in features_subsets:
        feature_names = ['SP_Ch1', 'SP_Ch2']
        names_list += feature_names
    names_list = list(dict.fromkeys(names_list))  # remove dups
    feature_len = len(names_list)
    # Note that len(names_list) > in_size, because of signal_name...
    return feature_len, names_list


def kc_template_matching(sw, fs, thresh):
    """
    template matching for seg
    :param sw: slow-wave seg from the signal
    :param fs: frequency
    :param thresh: Threshold from matching

    :return: True if matches to the template False otherwise
    """

    kc_template = pkl.load(open('kc_template.pkl', 'rb'))

    raise NotImplementedError


def compute_frequency_features(signal, fs, ch_names=['EEG', 'EEG(sec)'], normalize_signal=True):
    if normalize_signal:
        sscaler = StandardScaler()
        num_ch = signal.shape[0]
        signal = np.reshape(sscaler.fit_transform(np.reshape(signal, (-1, 1))), (num_ch, -1))
    return bandpower(signal, sf=fs, ch_names=ch_names, relative=True)[   # 'Gamma' removed
        ['Delta', 'Theta', 'Alpha', 'Beta']].mean().values


def compute_spindle_features(signal, fs, low=False, ch_names=['EEG', 'EEG(sec)']):
    if low:
        thresh = {'rel_pow': 0.1, 'corr': 0.5, 'rms': 0.8}  # low threshold config
    else:
        thresh = {'rel_pow': 0.3, 'corr': 0.68, 'rms': 1.2}  # default config (real default is rms=1.5, but ok)

    if len(ch_names) == 2:
        sp = spindles_detect_multi(signal, fs, ch_names=ch_names, multi_only=False, thresh=thresh)
    else:
        sp = spindles_detect(data=signal, sf=fs, thresh=thresh)
    spindle_features = np.zeros(7)
    if sp is not None:
        spindle_features = sp[['Amplitude', 'RMS', 'AbsPower', 'RelPower', 'Oscillations']].mean().values
        if len(ch_names) == 2:
            num_spindles_0 = np.array([sp.loc[sp.Channel == 'EEG', 'Channel'].shape[0]])
            num_spindles_1 = np.array([sp.loc[sp.Channel == 'EEG(sec)', 'Channel'].shape[0]])
        else:
            num_spindles_0 = np.array([sp.shape[0]])
            num_spindles_1 = 0
        spindle_features = np.append(spindle_features, np.append(num_spindles_0, num_spindles_1))
    return spindle_features


def compute_sw_features(signal, fs, ch_names=['EEG', 'EEG(sec)']):
    if len(ch_names) == 1:
        sw = sw_detect(signal, fs, hypno=None, include=(2, 3), freq_sw=(0.3, 3.5), dur_neg=(0.3, 1.5),
                       dur_pos=(0.1, 1), amp_neg=(40, 300), amp_pos=(10, 200), amp_ptp=(75, 500),
                       downsample=False, remove_outliers=False)
    else:
        sw = sw_detect_multi(signal, fs, ch_names=ch_names, hypno=None, include=(2, 3), freq_sw=(0.3, 3.5),
                             dur_neg=(0.3, 1.5), dur_pos=(0.1, 1), amp_neg=(40, 300),
                             amp_pos=(10, 200), amp_ptp=(75, 500), downsample=False,
                             remove_outliers=False)

    sw_features = np.zeros(8)
    if (sw is not None) and (len(sw) > 0):
        sw_features = sw[['Duration', 'ValNegPeak', 'ValPosPeak', 'PTP', 'Slope', 'Frequency']].values.mean(axis=0)
        if len(ch_names) == 2:
            num_sw_0 = np.array([sw.loc[sw.Channel == 'EEG', 'Channel'].shape[0]])
            num_sw_1 = np.array([sw.loc[sw.Channel == 'EEG(sec)', 'Channel'].shape[0]])
        else:
            num_sw_0 = np.array([sw.shape[0]])
            num_sw_1 = 0
        sw_features = np.append(sw_features, np.append(num_sw_0, num_sw_1))

    return sw_features


def normalize_features(features, save_name=None, save=False):
    x = features.values
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_features_norm = pd.DataFrame(x_scaled, columns=features.columns, index=features.index)
    df_features_norm = df_features_norm.reset_index()
    df_features_norm = df_features_norm.rename(columns={'index': 'signal_name'})
    if save:
        assert save_name is not None  # Must supply name to save!
        pd.to_pickle(df_features_norm,
                     os.path.join(os.getcwd(), 'shhs', 'preprocessed', 'features', save_name))
    return df_features_norm


def compute_features(signal, fs, features_to_compute, low, ch_names=['EEG', 'EEG(sec)']):
    features = []
    if 'frequency' in features_to_compute:
        features += [compute_frequency_features(signal, fs=fs, ch_names=ch_names, normalize_signal=True)]
    if 'spindle' in features_to_compute:
        temp = [compute_spindle_features(signal, low=low, fs=fs, ch_names=ch_names)]
        features += temp
    if 'sw' in features_to_compute:
        features += [compute_sw_features(signal, fs, ch_names=ch_names)]
    if len(features) > 0:
        features = np.hstack(features).astype('float32')
    return features
