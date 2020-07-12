import os
import torch
import numpy as np
import pandas as pd
import pickle as pkl
import pathlib
import matplotlib.pyplot as plt
from collections import Counter
from scipy.signal import savgol_filter, resample

from ECG.train.custom_dataloader import CustomDataset
from networks import HaifaNetVPT
from ECG.train.train_utils import get_device


def _fix_peaks(rr_local, waveform):
    for i, rpeak in enumerate(rr_local):
        qrs_interval = waveform[rpeak:rpeak + 12]
        rpeak = rpeak + np.argmax(qrs_interval)
        rr_local[i] = rpeak
    return rr_local


def get_numeric_template(cam_mat, before, fs=90):

    peak_loc = int(before*fs)
    pwave_start = peak_loc - int(0.25*fs)
    pwave_end = peak_loc - int(0.1*fs)
    s_peak = peak_loc + int(0.13*fs)
    t_peak_end = peak_loc + int(0.4*fs)

    pwave_activation = cam_mat[pwave_start:pwave_end, :].copy()
    qrs_activation = cam_mat[pwave_end:s_peak, :].copy()

    qr_activation = cam_mat[pwave_end:peak_loc, :].copy()
    rs_activation = cam_mat[peak_loc:s_peak, :].copy()
    st_activation = cam_mat[s_peak:t_peak_end, :].copy()

    mean_pwave = np.mean(np.mean(pwave_activation, axis=1))
    pwave_std = np.std(pwave_activation) / np.sqrt(np.size(pwave_activation))
    print(f'mean_pwave: {mean_pwave:.2f}, pwave std: {pwave_std:.5f}')

    qr_mean = np.mean(np.mean(qr_activation, axis=1))
    qr_std = np.std(qr_activation) / np.sqrt(np.size(qr_activation))
    print(f'qr_mean: {qr_mean:.2f}, qr std: {qr_std:.5f}')

    rs_mean = np.mean(np.mean(rs_activation, axis=1))
    rs_std = np.std(rs_activation) / np.sqrt(np.size(rs_activation))
    print(f'rs_mean: {rs_mean:.2f}, qr std: {rs_std:.5f}')

    st_mean = np.mean(np.mean(st_activation, axis=1))
    st_std = np.std(st_activation) / np.sqrt(np.size(st_activation))
    print(f'st_mean: {st_mean:.2f}, qr std: {st_std:.5f}')

    qrs_mean = np.mean(np.mean(qrs_activation, axis=1))
    qrs_std = np.std(qrs_activation) / np.sqrt(np.size(qrs_activation))
    print(f'qrs_mean: {qrs_mean:.2f}, qr std: {qrs_std:.5f}')

    return


def _get_templates(waveform, rpeaks, before, after, fs, idx1=0, idx2=5400, noise_lim_sec=0):

    # This is from class_activation_map_template.py

    # convert delimiters to samples
    before = int(before * fs)
    after = int(after * fs)
    noise_lim_samples = noise_lim_sec * fs  # no need to convert to int here, this will be done later

    # Sort R-Peaks in ascending order
    rpeaks = np.sort(rpeaks)

    # Get number of sample points in waveform
    length = len(waveform)

    # Create empty list for templates
    templates = []

    # Create empty list for new rpeaks that match templates dimension
    rpeaks_new = np.empty(0, dtype=int)

    # Loop through R-Peaks
    rpeaks += np.random.uniform(low=-noise_lim_samples, high=noise_lim_samples, size=rpeaks.shape).astype(int)
    for rpeak in rpeaks:

        # Before R-Peak
        a = rpeak - before
        if a < 0:
            continue

        # After R-Peak
        b = rpeak + after
        if b - idx1 > length:
            break

        if (rpeak > idx1) & (rpeak < idx2):
            # Append template list
            templates.append(waveform[a:b])

            # Append new rpeaks list
            rpeaks_new = np.append(rpeaks_new, rpeak)

    # Convert list to numpy array
    templates = np.array(templates).T

    return templates, rpeaks_new


def plot_cam(saved_model_name, signal_name, plot_inds, test_loader, model, cam_target, label='normal',
             use_grad_cam=False, use_relu=True, grad_weight=True, ds=True):
    # The original function included also a plot of the 'templete' activation.
    # We removed it for the time being, but it can be found in Seb's original code
    label_lookup = ['Normal Sinus Rhythm', 'Atrial Fibrillation', 'Other Rhythm', 'Noisy']
    # if use_grad_cam:
    #     conv0_len = 9000

    if ds:
        fs = 90  # [Hz]
        signal_len = 5400
        conv0_len = 2700
    else:
        fs = 300  # [Hz]
        signal_len = 18000
        conv0_len = 2250

    signal_index = int(test_loader.dataset.labels.loc[test_loader.dataset.labels.signal == signal_name].index.values)

    time_series, true_label, _, _, _, feature_rep = test_loader.dataset[signal_index]
    time_series = time_series.reshape(signal_len)
    time_series_tensor = torch.reshape(torch.tensor(time_series), (1, 1, signal_len)).to("cpu")
    feature_tensor = torch.reshape(torch.tensor(feature_rep), (1, -1)).to("cpu")
    logits, cam, _ = model(time_series_tensor, feature_tensor)

    if use_grad_cam:
        logits[:, cam_target].backward()
        gradients = model.get_activations_gradient()
        pooled_gradients = torch.mean(gradients, dim=[0, 2])
        activations = model.get_activations().detach()
        if grad_weight:  # weight by gradients
            for i in range(activations.shape[1]):  # change to torch.mm later
                activations[:, i, :] *= pooled_gradients[i]
        grad_cam = torch.mean(activations, dim=1).squeeze()
        if use_relu:
            grad_cam = np.maximum(grad_cam, 0)
            grad_cam /= torch.max(grad_cam)
        else:
            grad_cam = (grad_cam - torch.min(grad_cam)) / (torch.max(grad_cam) - torch.min(grad_cam))

        cam = grad_cam.numpy()
    else:
        cam = np.squeeze(cam.detach().numpy())[cam_target, :]
        # Clip signal to baseline
        cam = np.maximum(cam, Counter(cam).most_common(1)[0][0])
        # cam = (cam - min(cam)) / (max(cam) - min(cam))

    logits = np.squeeze(logits.detach().numpy())

    # The following is from Seb's plot_class_activation_map and plot_class_activation_map_template
    cam_time = np.arange(conv0_len) / (conv0_len / 60)
    non_zero_index = np.where(abs(time_series) > 1e-4)[0]  # The downsampling process introduced artifacts so 0 is not really 0, more like 1e-5

    time_series_filt = time_series[non_zero_index[0]:non_zero_index[-1]]
    time_series_filt_ts = np.arange(time_series_filt.shape[0]) * 1 / fs

    cam_time_intrp = np.arange(time_series.shape[0]) * 1 / fs
    cam_intrp = np.interp(cam_time_intrp, cam_time, cam)

    cam_filt = cam_intrp[non_zero_index[0]:non_zero_index[-1]]
    prob = np.exp(logits) / sum(np.exp(logits))

    # Setup figure
    fig = plt.figure(figsize=(15, 10))
    fig.subplots_adjust(wspace=0, hspace=0)
    ax1 = plt.subplot2grid((2, 5), (0, 0), colspan=5)
    ax2 = plt.subplot2grid((2, 5), (1, 0), colspan=5)
    # ax3 = plt.subplot2grid((3, 5), (2, 1), colspan=3)

    # Set plot title
    if 'baseline' in saved_model_name.lower():
        model_type = 'Baseline -'
        featureset = ''
    else:
        model_type = 'HSIC - '
        featureset = 'Given rr - ' if 'rr' in saved_model_name else 'Given all -'
    ax1.set_title(model_type + featureset +
        signal_name + '- True Label: ' + label_lookup[np.squeeze(true_label)] + '\n' +
        'Predicted Label: ' + label_lookup[np.squeeze(np.argmax(logits))] + '\n' +
        'Normal Sinus Rhythm: ' + str(np.round(prob[0], 2)) +
        '     Atrial Fibrillation: ' + str(np.round(prob[1], 2)) +
        '     Other Rhythm: ' + str(np.round(prob[2], 2)),
        fontsize=20, y=1.03
    )

    idx1 = plot_inds[0]
    idx2 = plot_inds[1] if plot_inds[1] < time_series_filt_ts.shape[0] else time_series_filt_ts.shape[0]-1

    # Plot image
    ax1.plot(time_series_filt_ts[idx1:idx2], time_series_filt[idx1:idx2], '-k', lw=1.5)

    # Axes labels
    ax1.set_ylabel('Normalized Amplitude', fontsize=22)
    ax1.set_xlim([time_series_filt_ts[idx1], time_series_filt_ts[idx2].max()])
    ax1.tick_params(labelbottom='off')
    ax1.yaxis.set_tick_params(labelsize=16)

    # Plot CAM
    ax2.plot(time_series_filt_ts[idx1:idx2], cam_filt[idx1:idx2], '-k', lw=1.5)
    # Axes labels
    ax2.set_xlabel('Time, seconds', fontsize=22)
    ax2.set_ylabel('Class Activation Map', fontsize=22)
    ax2.set_xlim([time_series_filt_ts[idx1], time_series_filt_ts[idx2].max()])
    # ax2.set_ylim([cam_filt.min()-0.05, cam_filt.max()+0.05])
    # ax2.set_ylim([-3, 35])
    ax2.xaxis.set_tick_params(labelsize=16)
    ax2.yaxis.set_tick_params(labelsize=16)

    plt.show()

    cam_template = True
    if cam_template:

        main_dir = os.path.abspath(os.path.join(pathlib.Path(__file__).parent.absolute(), os.pardir))
        rr_locals = pd.read_csv(os.path.join(main_dir, 'data', 'rr_local', 'rr_locals_90_test_filtered7.csv'))
        rr_locals.set_index('Unnamed: 0', inplace=True)
        rr_local = rr_locals.loc[signal_name].values.astype(int)
        if (rr_local != 0).any():
            templates, _ = _get_templates(cam_filt, rr_local, before=0.3, after=0.1, fs=90)
            plot_cam_template(templates, name=signal_name, before=0.3, after=0.1)
        else:
            print(' rr_local is all 0')


def plot_cam_template(file_name, noise_lim_sec=0.):

    # Dirty patch..first is for running from this script, second is for running from one folder up
    try:
        with open(os.path.join(os.getcwd(), '..', 'saved_models', file_name, f'figure_data_noise={noise_lim_sec}.pkl'), 'rb') as h:
            save_data = pkl.load(h)
    except:
        with open(os.path.join(os.getcwd(), 'saved_models', file_name, f'figure_data_noise={noise_lim_sec}.pkl'), 'rb') as h:
            save_data = pkl.load(h)

    templates = save_data['cam_mat']
    templates_sig = save_data['sig_mat']
    before = save_data['before']
    after = save_data['after']

    get_numeric_template(templates, before=before, fs=90)
    print(f"total activation mean: {save_data['total_mean_activation']:.2f}, "
          f"total activation std: {save_data['total_activation_std']:.5f}")

    templates_ts = np.linspace(-before*1000, after*1000, templates.shape[0], endpoint=False)
    fig, ax = plt.subplots(1, 1)
    mean_cam = np.mean(templates, axis=1)
    mean_cam = savgol_filter(mean_cam, 11, 2)

    if templates_sig is not None:
        mean_sig = .5*np.mean(templates_sig, axis=1)+0.2
        ax.plot(templates_ts, mean_sig, 'grey', linewidth=5, alpha=0.5, label='ECG')

    ax.plot(templates_ts, mean_cam, '-k', label='Activation')
    ax.set_xlabel('Time [ms]', fontsize=20)
    ax.set_ylabel('Amplitude [ms]', fontsize=20)
    ax.set_ylim([0, 0.8])

    # Plot R peak
    # plt.axvline(0, color='red', linestyle='--')
    # ax.text(0.75, 0.5, 'R-peak', fontdict={'size': 13, 'style': 'italic', 'color': 'red'});
    # ax.set_xticks(np.arange(-before * 100, after * 100, 10))

    # plot p-wave window
    qrs_start = -80
    ax.axvspan(-250, qrs_start, alpha=0.3, color='xkcd:medium purple', label='P-Wave Window')

    # plot QRS window
    ax.axvspan(qrs_start, 65, alpha=0.3, color='teal', label='QRS Complex')

    plt.grid()
    plt.legend()
    plt.show()
    return


def get_global_template(file_name, cam_target=1, label='af', feature_subset='rr',
                        feature_opt='HSIC+Concat', noise_lim_sec=0.):
    cuda_id = 0
    device = get_device(cuda_id)
    label_lookup = ['Normal Sinus Rhythm', 'Atrial Fibrillation', 'Other Rhythm', 'Noisy']
    fs = 90  # [Hz]
    signal_len = 5400
    conv0_len = 2700

    test_dataset = CustomDataset("test", feature_subset=feature_subset,
                                 feature_opt=feature_opt, oversample=label)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    model = HaifaNetVPT(num_classes=3, feature_len=test_dataset.feature_len, feature_opt=feature_opt, gap_norm_opt='batch_norm')

    main_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    model_to_load = f'{file_name}_params.pkl'

    model.load_state_dict(torch.load(os.path.join(main_dir, 'saved_models', file_name, model_to_load), map_location='cpu'))
    model.eval()

    rr_locals = pd.read_csv(os.path.join(main_dir, 'data', 'rr_local', 'rr_locals_90_test_filtered7.csv'))
    rr_locals.set_index('Unnamed: 0', inplace=True)

    all_templates_cam = []
    all_templates_sig = []
    total_activation = []
    list_signal_names = []
    before = 0.7
    after = 0.7

    for batch_idx, (data, target, feature, _, signal_name, feature_rep) in enumerate(test_loader):
        logits, cam, _ = model(data, feature_rep)

        cam = np.squeeze(cam.detach().numpy())[cam_target, :]
        cam = np.maximum(cam, Counter(cam).most_common(1)[0][0])
        cam = resample(cam, signal_len)

        data = np.squeeze(data.detach().numpy())

        rr_local = rr_locals.loc[signal_name].values.astype(int)
        rr_local = _fix_peaks(rr_local, data)
        rr_local = rr_local[rr_local > 1]
        if (rr_local != 0).any():

            tmpls_cam, rr1 = _get_templates(cam, rr_local, before=before, after=after, fs=90, noise_lim_sec=noise_lim_sec)
            if tmpls_cam.shape[0]:
                all_templates_cam.append(tmpls_cam)
                list_signal_names.append([f'{signal_name}_{i}' for i in range(tmpls_cam.shape[1])])

            tmpls_sig, rr2 = _get_templates(data, rr_local, before=before, after=after, fs=90, noise_lim_sec=0)
            if tmpls_sig.shape[0]:
                all_templates_sig.append(tmpls_sig)

        non_zero_index = np.where(abs(data) > 1e-4)[0]  # The downsampling process introduced artifacts so 0 is not really 0, more like 1e-5
        total_activation.append(cam[non_zero_index[0]:non_zero_index[-1]])

    cam_mat = np.hstack(all_templates_cam)
    sig_mat = np.hstack(all_templates_sig)

    for i in range(cam_mat.shape[1]):
        cam_i = cam_mat[:, i]
        if ((cam_i-cam_i.mean()) != 0).sum() < 5:
            continue
        if max(cam_i) != min(cam_i):
            cam_i = (cam_i - min(cam_i)) / (max(cam_i) - min(cam_i))
            cam_mat[:, i] = cam_i

    # generate numeric measures
    total_mean_activation = np.mean([np.mean(total_activation[i]) for i in range(len(total_activation))])
    total_activation_std = np.std(np.hstack(total_activation)) / np.sqrt(np.size(np.hstack(total_activation)))

    save_data = {'total_mean_activation': total_mean_activation,
                 'total_activation_std': total_activation_std,
                 'cam_mat': cam_mat,
                 'sig_mat': sig_mat,
                 'before': before,
                 'after': after,
                 'list_signal_names': list_signal_names}

    with open(os.path.join(os.getcwd(), os.pardir, 'saved_models', file_name, f'figure_data_noise={noise_lim_sec}_pval.pkl'), 'wb') as h:
        pkl.dump(save_data, h, protocol=pkl.HIGHEST_PROTOCOL)
    return cam_mat, sig_mat, before, after


if __name__ == "__main__":
    file_name = 'lambda_500_pwave_naf'
    noise_lim_sec = 0.  # sec
    cam_mat, sig_mat, before, after = get_global_template(file_name=file_name, feature_subset='p_wave',
                                                          feature_opt='HSIC+Concat', noise_lim_sec=noise_lim_sec)
    plot_cam_template(file_name=file_name, noise_lim_sec=noise_lim_sec)
