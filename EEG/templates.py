import os
import torch
import pickle as pkl
from yasa import spindles_detect_multi, get_bool_vector, spindles_detect, sw_detect, sw_detect_multi, rem_detect
import mne
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import find_peaks
from biosppy.signals.emg import silva_onset_detector

from networks import HSICClassifier
from EEG.datasets import init_datasets
from EEG.train_utils import get_device, run_params


def events_to_index(x):
    sp = np.split(x, np.where(np.diff(x) != 1)[0] + 1)
    return np.array([[k[0], k[-1]] for k in sp]).astype(int)


def _index_to_events(x):
    index = np.array([])
    for k in range(x.shape[0]):
        index = np.append(index, np.arange(x[k, 0], x[k, 1] + 1))
    return index.astype(int)


def get_bool_vector(data=None, sf=None, detection=None):
    if isinstance(data, mne.io.BaseRaw):
        sf = data.info['sfreq']  # Extract sampling frequency
        data = data.get_data() * 1e6  # Convert from V to uV
        data = np.squeeze(data)  # Flatten if only one channel is present

    data = np.asarray(data)
    assert isinstance(detection, pd.DataFrame)
    assert 'Start' in detection.keys()
    assert 'End' in detection.keys()
    bool_vector = np.zeros(data.shape, dtype=int)

    # For multi-channel detection
    multi = False
    if (data.shape[0] > 1) and ('Channel' in detection.keys()):
        chan = detection['Channel'].unique()
        multi = True
    if multi:
        for c in chan:
            sp_chan = detection[detection['Channel'] == c]
            idx_sp = _index_to_events(sp_chan[['Start', 'End']].values * sf)
            bool_vector[sp_chan['IdxChannel'].iloc[0], idx_sp] = 1
    else:
        idx_sp = _index_to_events(detection[['Start', 'End']].values * sf)
        try:
            bool_vector[idx_sp] = 1
        except IndexError:
            print('hi')
    return bool_vector


def replace_starts_with_random(starts, random, sec_before, sec_after, fs, signal_len, noise_inter=0):
    if random:
        # generate a random sample only for detections that are valid in terms of window size.
        # starts values with invalid indices will anyway be
        # filtered by indices_to_templates so no need to generate for them.
        valid_indices = ((starts > sec_before*fs) & (starts < signal_len - sec_after * fs))
        num_valid_detections = valid_indices.sum()

        noise_lim_samples = noise_inter * fs
        high = starts[valid_indices] + noise_lim_samples
        low = starts[valid_indices] - noise_lim_samples
        print(np.random.uniform(low=low, high=high, size=num_valid_detections))
        starts[valid_indices] = np.random.uniform(low=low, high=high,
                                                  size=num_valid_detections).astype(int)
    return starts


def get_spindle_templates(signal, cam=None, fs=125, sec_before=5., sec_after=5., num_ch=2, random=False, noise_inter=0):
    cam_time_intrp = np.arange(signal.shape[1]) * 1 / fs
    cam_intrp = np.interp(cam_time_intrp, np.arange(cam.shape[0]) / (cam.shape[0] / (signal.shape[1] / fs)), cam)
    assert(cam_intrp.shape[0] == 2400)  # just checking
    cam_templates = []
    # thresh = {'rel_pow': 0.1, 'corr': 0.5, 'rms': 0.8}
    thresh = {'rel_pow': 0.3, 'corr': 0.68, 'rms': 1.5}

    if num_ch == 1:
        sp = spindles_detect(signal, fs, thresh=thresh)
    else:
        sp = spindles_detect_multi(signal, fs, multi_only=False, thresh=thresh, ch_names=['EEG', 'EEG(sec)'])

    if sp is not None:
        sp_starts = df_to_event_start(df=sp, signal=signal, fs=fs, num_ch=num_ch)

        sp_starts = replace_starts_with_random(random=random, starts=sp_starts, sec_before=sec_before,
                                               sec_after=sec_after, fs=fs, signal_len=cam_intrp.shape[0],noise_inter=noise_inter)
        cam_templates = indices_to_templates(cam_intrp, indices=sp_starts, sec_before=sec_before, sec_after=sec_after, fs=fs)

    return cam_templates


def get_sw_templates(cam, signal, sec_before, sec_after, fs, num_ch, random=False,noise_inter=0):
    cam_time_intrp = np.arange(signal.shape[1]) * 1 / fs
    cam_intrp = np.interp(cam_time_intrp, np.arange(cam.shape[0]) / (cam.shape[0] / (signal.shape[1] / fs)), cam)
    assert(cam_intrp.shape[0] == 2400)  # just checking
    # num_ch = 1
    # # using the first channel only
    # signal = signal[0]
    templates = []
    if num_ch == 1:
        sw = sw_detect(signal, fs, hypno=None, include=(2, 3), freq_sw=(0.3, 3.5), dur_neg=(0.3, 1.5),
                       dur_pos=(0.1, 1), amp_neg=(40, 300), amp_pos=(10, 200), amp_ptp=(75, 500),
                       downsample=False, remove_outliers=False)
    else:
        sw = sw_detect_multi(signal, fs, ch_names=['EEG', 'EEG(sec)'], hypno=None, include=(2, 3), freq_sw=(0.3, 3.5),
                             dur_neg=(0.3, 1.5), dur_pos=(0.1, 1), amp_neg=(40, 300),
                             amp_pos=(10, 200), amp_ptp=(75, 500), downsample=False,
                             remove_outliers=False)

    if (sw is not None) and (len(sw) > 0):
        sw_starts = df_to_event_start(df=sw, signal=signal, fs=fs, num_ch=num_ch)
        sw_starts = replace_starts_with_random(random=random, starts=sw_starts, sec_before=sec_before,
                                               sec_after=sec_after, fs=fs, signal_len=cam_intrp.shape[0], noise_inter=noise_inter)
        templates = indices_to_templates(cam_intrp, indices=sw_starts, sec_before=sec_before, sec_after=sec_after, fs=fs)

    return templates


def get_rem_templates(cam, signal_eog, sec_before, sec_after, fs_eeg, random=False,noise_inter=0):

    templates = []
    loc = signal_eog[1, :]
    roc = signal_eog[0, :]
    fs_eog = 50
    cam_time_intrp = np.arange(signal_eog.shape[1]) * 1 / fs_eog
    cam_intrp = np.interp(cam_time_intrp, np.arange(cam.shape[0]) / (cam.shape[0] / (signal_eog.shape[1] / fs_eog)), cam)

    rems = rem_detect(loc, roc, sf=fs_eog, hypno=None, include=4, amplitude=(50, 325), duration=(0.3, 1.5),
                      freq_rem=(0.5, 5), downsample=True, remove_outliers=False)

    if (rems is not None) and (len(rems) > 0):
        rem_starts = df_to_event_start(df=rems, signal=cam_intrp, fs=fs_eog, num_ch=1)
        rem_starts = replace_starts_with_random(random=random, starts=rem_starts, sec_before=sec_before,
                                                sec_after=sec_after, fs=fs_eog, signal_len=cam_intrp.shape[0], noise_inter=noise_inter)
        templates = indices_to_templates(cam_intrp, indices=rem_starts, sec_before=sec_before, sec_after=sec_after, fs=fs_eeg)

    return templates


def get_emg_onset_templates(cam, signal_emg, sec_before, sec_after, fs_eeg=125, random=False):
    fs_emg = 125
    cam_time_intrp = np.arange(signal_emg.shape[0]) * 1 / fs_emg
    cam_intrp = np.interp(cam_time_intrp, np.arange(cam.shape[0]) / (cam.shape[0] / (signal_emg.shape[0] / fs_emg)), cam)

    # ts, filtered, onsets = emg(signal=signal_emg, sampling_rate=fs_emg, show=False)
    onsets, processed = silva_onset_detector(signal=signal_emg, sampling_rate=fs_emg, size=1, threshold_size=2, threshold=10)
    templates = []
    if onsets.size > 0:
        window_sec = 1
        filtered_onsets = filter_peaks_distance(signal=signal_emg, peaks=onsets, window_sec=window_sec, fs=fs_emg)
        filtered_onsets = replace_starts_with_random(random=random, starts=filtered_onsets, sec_before=sec_before,
                                                     sec_after=sec_after, fs=fs_emg, signal_len=cam_intrp.shape[0])
        templates = indices_to_templates(cam_intrp, indices=filtered_onsets, sec_before=sec_before, sec_after=sec_after, fs=fs_eeg)
    return templates


def filter_peaks_distance(signal, peaks, window_sec, fs):
    peaks = peaks.astype(int)
    onsets_mask = np.zeros(signal.shape)
    onsets_mask[peaks] = np.flipud(np.arange(peaks.shape[0]))
    filtered_onsets = find_peaks(onsets_mask, threshold=0.5, distance=window_sec * fs)[0]
    return filtered_onsets


def get_activation_templates(cam, signal, sec_before, sec_after, fs):
    cam_time_intrp = np.arange(signal.shape[1]) * 1 / fs
    cam_intrp = np.interp(cam_time_intrp, np.arange(cam.shape[0]) / (cam.shape[0] / (signal.shape[1] / fs)), cam)
    peaks_inds, _ = find_peaks(cam_intrp, prominence=10, distance=int((sec_before+sec_after)*fs))
    templates = indices_to_templates(signal, indices=peaks_inds, sec_before=sec_before, sec_after=sec_after, fs=fs)
    return templates


def df_to_event_start(df, signal, fs, num_ch):
    bool_vector = get_bool_vector(signal, fs, df)
    if num_ch == 2:
        bool_vector = np.logical_or(bool_vector[0, :], bool_vector[1, :]).astype(int)
    event_idx = np.where(bool_vector)[0]
    event_starts, _ = events_to_index(event_idx).T
    return event_starts


def indices_to_templates(signal, indices, sec_before, sec_after, fs):
    samp_before = int(sec_before * fs)
    samp_after = int(sec_after * fs)
    templates = []
    for s in indices:
        temp = signal[(s - samp_before):(s + samp_after)]
        if samp_before + samp_after == temp.shape[0]:
            # check that template is not clipped by start or end of sig
            templates += [signal[(s - samp_before):(s + samp_after)]]
    return templates


def get_templates(file_name, oversample, template_opt='spindle', cam_target=2, num_patients=40, num_ch=2, before=5., after=5.,
                  task='all', try2load=True, gap_norm_opt='batch_norm', cuda_id=0, random=False, normalize_signals=False, noise_inter=0):
    device = get_device(cuda_id)
    features_subset = []
    if 'freq' in file_name.lower():
        features_subset += ['frequency']
    if 'num_spindles' in file_name.lower():
        features_subset += ['num_spindles']
    elif 'spindle' in file_name.lower():
        features_subset += ['spindle']
    assert template_opt in ['spindle', 'activation', 'sw', 'rem', 'emg']

    low_sp = 'low' in file_name
    feature_opt, signal_len, one_slice, dataset_dir = run_params(file_name, features_subset, def_feature_opt='HSIC+Concat',
                                                                 task=task)
    fs = 80 if 'ds' in file_name.lower() else 125
    file_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_models', file_name)
    num_classes = 2 if 'rem' in task.lower() else 5

    print(f'Sample Frequency: {fs}')
    print(f'Num Classes: {num_classes}')
    random_str = 'random_' if random else ''
    save_name = f'{random_str}{template_opt}_template_class{cam_target}_{file_name}_{num_patients}patients'
    file_path = os.path.join('plot_results', save_name)
    
    if try2load and os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            all_templates = pkl.load(f)
    else:
        filter_stage = cam_target if isinstance(cam_target, int) else None
        test_loader = init_datasets(num_patients=num_patients, dataset_dir=dataset_dir, batch_size=1, conv_d=1,
                                    features_subset=features_subset, one_slice=one_slice, modes=['test'],
                                    random_flag=False, num_ch=num_ch, low_sp=low_sp, filter_stage=filter_stage,
                                    task=task, normalize_signals=normalize_signals, oversample=oversample)
    
        model = HSICClassifier(num_classes=num_classes, signal_len=signal_len, feature_opt=feature_opt,
                               in_channels=num_ch, feature_len=test_loader.dataset.feature_len, gap_norm_opt=gap_norm_opt)
        model.load_state_dict(torch.load(os.path.join(file_dir, f'{file_name}_params.pkl'), map_location='cpu'))
        model.to(device)
        model.eval()
    
        all_templates = []
    
        with torch.no_grad():
            for batch_idx, (signal, label, signal_name, features) in enumerate(tqdm(test_loader)):
                signal, features = signal.to(device), features.to(device)
                _, cam, _ = model(signal, features)

                signal = test_loader.dataset.unnormalize_signal(signal)
                signal = signal.cpu().numpy().reshape(-1, signal.shape[-1])
                cam = np.squeeze(cam).cpu().numpy()
                if isinstance(cam_target, int):
                    cam = cam[cam_target, :]
                else:
                    cam = cam[label, :]
    
                signal = np.squeeze(signal)
                if template_opt == 'spindle':
                    templates = get_spindle_templates(signal, fs=fs, cam=cam, random=random,
                                                      sec_before=before, sec_after=after, num_ch=num_ch, noise_inter=noise_inter)
                if template_opt == 'activation':
                    if num_ch == 2:
                        signal = signal[0, :]  # TODO!!
                    templates = get_activation_templates(cam, signal, sec_before=before, sec_after=after, fs=fs)
                if template_opt == 'sw':
                    templates = get_sw_templates(cam, signal, sec_before=before, sec_after=after, fs=fs,
                                                 num_ch=num_ch, random=random, noise_inter=noise_inter)
                if template_opt == 'rem':
                    eog = test_loader.dataset.get_eog(signal_name)
                    templates = get_rem_templates(cam=cam, signal_eog=eog, sec_before=before, sec_after=after,
                                                  fs_eeg=fs, random=random, noise_inter=noise_inter)
                if template_opt == 'emg':
                    signal_emg = test_loader.dataset.get_emg(signal_name).squeeze()
                    templates = get_emg_onset_templates(cam=cam, signal_emg=signal_emg, sec_before=before,
                                                        sec_after=after, fs_eeg=fs, random=random)


                all_templates += templates
    
        all_templates = np.vstack(all_templates).T

        # num_templates = all_templates.shape[0]
        # Normalize templates
        # for i in range(num_templates):
        #     cam_i = all_templates[:, i]
        #     if ((cam_i - cam_i.mean()) != 0).sum() < 5:
        #         continue
        #     if max(cam_i) != min(cam_i):
        #         cam_i = (cam_i - min(cam_i)) / (max(cam_i) - min(cam_i))
        #         all_templates[:, i] = cam_i

        if random is not None:
            save_name = os.path.join('noise', f'{save_name}_figure_data_noise={noise_inter}')

        with open(os.path.join('plot_results', save_name), 'wb') as f:
            pkl.dump(all_templates, f, protocol=pkl.HIGHEST_PROTOCOL)
    return all_templates
