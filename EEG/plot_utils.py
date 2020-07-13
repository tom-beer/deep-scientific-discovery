import torch
import numpy as np
import matplotlib.pyplot as plt
from yasa import get_bool_vector, spindles_detect
from EEG.templates import get_templates
import pandas as pd
import pickle as pkl
import glob


def plot_cam(saved_model_name, signal_name, plot_inds, test_loader, model, cam_target, label='normal',
             use_grad_cam=False, use_relu=True, grad_weight=True, ds=False):

    label_lookup = ['Wake', 'N1', 'N2', 'N3', 'REM']

    if ds:
        fs = 80  # [Hz]
        signal_len = 2400
        conv0_len = 2700  # signal_len / 2
    else:
        fs = 125  # [Hz]
        signal_len = 15000
        conv0_len = 1875  # signal_len/(2**3)

    time_series, true_label, name = test_loader.dataset[0]
    time_series_tensor = torch.reshape(torch.tensor(time_series), (1, 2, signal_len)).to("cpu")
    raise NotImplementedError  # this is not updated..
    feature_tensor = torch.zeros(1)  # todo: fix for real features
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
        # cam = np.maximum(cam, Counter(cam).most_common(1)[0][0])
        # cam = (cam - min(cam)) / (max(cam) - min(cam))

    logits = np.squeeze(logits.detach().numpy())

    # The following is from Seb's plot_class_activation_map and plot_class_activation_map_template

    cam_time = np.arange(cam.shape[0]) / (cam.shape[0] / 120)
    cam_time_intrp = np.arange(time_series.shape[1]) * 1 / fs
    cam_intrp = np.interp(cam_time_intrp, cam_time, cam)

    # relevant slice
    before = 60
    after = 30
    # time_series = time_series[:, int(before*fs): int(-after*fs)]
    # cam_intrp = cam_intrp[int(before*fs):int(-after*fs)]

    time_series_filt = time_series
    time_series_filt_ts = np.arange(time_series_filt.shape[1]) * 1 / fs

    cam_filt = cam_intrp
    prob = np.exp(logits) / sum(np.exp(logits))

    sp1 = spindles_detect(time_series_filt[0, :], fs)
    if sp1 is not None:
        bool_spindles1 = get_bool_vector(time_series_filt[0, :], fs, sp1)
        spindles_highlight1 = time_series_filt[0, :] * bool_spindles1
        spindles_highlight1[spindles_highlight1 == 0] = np.nan
        spindles_highlight1 = spindles_highlight1[:-1]

    sp2 = spindles_detect(time_series_filt[1, :], fs)
    if sp2 is not None:
        bool_spindles2 = get_bool_vector(time_series_filt[1, :], fs, sp2)
        spindles_highlight2 = time_series_filt[1, :] * bool_spindles2
        spindles_highlight2[spindles_highlight2 == 0] = np.nan
        spindles_highlight2 = spindles_highlight2[:-1]

    # plt.figure(figsize=(14, 4))
    # plt.plot(times, data, 'k')
    # plt.plot(times, spindles_highlight, 'indianred')

    # Setup figure
    fig = plt.figure(figsize=(15, 10))
    fig.subplots_adjust(wspace=0, hspace=0)
    ax1 = plt.subplot2grid((3, 5), (0, 0), colspan=5)
    ax2 = plt.subplot2grid((3, 5), (1, 0), colspan=5)
    ax3 = plt.subplot2grid((3, 5), (2, 0), colspan=5)

    class_target_dict = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
    title = f'{signal_name}'

    ax1.set_title(title + '\n'
                          f'Truth: {label}' + '\n' +
                          f'Predicted Label: {label_lookup[np.squeeze(np.argmax(logits))]}  ' +
                          str(np.round(prob[np.squeeze(np.argmax(logits))], 2)),
                          fontsize=20, y=1.03
                  )

    idx1 = plot_inds[0]
    idx2 = plot_inds[1] if plot_inds[1] < time_series_filt_ts.shape[0] else time_series_filt_ts.shape[0]-1

    # Plot image
    ax1.plot(time_series_filt_ts[idx1:idx2], time_series_filt[0, idx1:idx2], '-k', lw=1.5)
    if sp1 is not None:
        ax1.plot(time_series_filt_ts[idx1:idx2], spindles_highlight1, 'indianred')
    ax1.set_ylabel('Normalized Amplitude', fontsize=22)
    ax1.set_xlim([time_series_filt_ts[idx1], time_series_filt_ts[idx2].max()])
    ax1.tick_params(labelbottom='off')
    ax1.yaxis.set_tick_params(labelsize=16)

    ax2.plot(time_series_filt_ts[idx1:idx2], time_series_filt[1, idx1:idx2], '-k', lw=1.5)
    if sp2 is not None:
        ax2.plot(time_series_filt_ts[idx1:idx2], spindles_highlight2, 'indianred')
    ax2.set_ylabel('Normalized Amplitude', fontsize=22)
    ax2.set_xlim([time_series_filt_ts[idx1], time_series_filt_ts[idx2].max()])
    ax2.tick_params(labelbottom='off')
    ax2.yaxis.set_tick_params(labelsize=16)

    # Plot CAM
    ax3.plot(time_series_filt_ts[idx1:idx2], cam_filt[idx1:idx2], '-k', lw=1.5)
    ax3.set_xlabel('Time, seconds', fontsize=22)
    ax3.set_ylabel('Class Activation Map', fontsize=22)
    ax3.set_xlim([time_series_filt_ts[idx1], time_series_filt_ts[idx2].max()])
    # ax2.set_ylim([cam_filt.min()-0.05, cam_filt.max()+0.05])
    ax3.xaxis.set_tick_params(labelsize=16)
    ax3.yaxis.set_tick_params(labelsize=16)

    plt.show()


def extract_results(file_name):
    try:
        with open(f'saved_models/{file_name}/{file_name}_test_perf.pkl', 'rb') as f:
            classi_perf = pkl.load(f)
        print("Classification Performance:")
        print(f'Accuracy: {classi_perf["accuracy"]:.2f}, Kappa: {classi_perf["kappa_score"]:.2f}, '
              f'f1m: {classi_perf["f1m"]:.2f}, f1M: {classi_perf["f1M"]:.2f}')
        print("")
    except FileNotFoundError:
        print("Warning: no test_perf.pkl found in folder")
    
    try:
        with open(f'saved_models/{file_name}/{file_name}_test_r2.pkl', 'rb') as f:
            r2 = pkl.load(f)
        print("Independence results")
        print(f"R2: {r2:.3f}")
        print("")
    except FileNotFoundError:
        print("Warning: no test_r2.pkl found in folder")
    try:
        with open(f'saved_models/{file_name}/{file_name}_rep2label_perf.pkl', 'rb') as f:
            rep2label_perf = pkl.load(f)
        s_rep = pd.DataFrame(rep2label_perf).mean()  # average all cross validation folds
        print("Rep2Label Performance:")
        print(f'Accuracy: {s_rep["accuracy"]:.2f}, Kappa: {s_rep["kappa_score"]:.2f}, '
              f'f1m: {s_rep["f1m"]:.2f}, f1M: {s_rep["f1M"]:.2f}')
        print("")
        
    except FileNotFoundError:
        print("Warning: no rep2label_perf.pkl found in folder")   

    return


def plot_templates(file_name, cam_target, text_height, before=5., after=5., template_opt='spindle', task='all',
                   num_patients=None, random=False):
    # If running from notebook, it is advisable to leave num_patients None, and so the highest number of patients will be loaded
    assert (template_opt in ['rem', 'emg', 'sw', 'spindle'])
    random_str = 'random_' if random else ''
    if num_patients is None:
        # Load saved templates (find the highest number of patients)
        list_of_templates = glob.glob(f'plot_results/{random_str}{template_opt}_template_class{cam_target}_{file_name}*')
        list_of_num_patients = [int(file.split('patients')[0].split('_')[-1]) for file in list_of_templates]
        best_file_idx = np.argmax(list_of_num_patients)
        num_patients = list_of_num_patients[best_file_idx]
        template_file_name = list_of_templates[best_file_idx]
    else:
        template_file_name = f'plot_results/{random_str}{template_opt}_template_class{cam_target}_{file_name}_' \
                             f'{num_patients}patients_'

    with open(template_file_name, 'rb') as f:
        templates = pkl.load(f)
    num_templates = templates.shape[1]
    # Normalize templates
    for i in range(num_templates):
        cam_i = templates[:, i]
        if ((cam_i - cam_i.mean()) != 0).sum() < 5:
            continue
        if max(cam_i) != min(cam_i):
            cam_i = (cam_i - min(cam_i)) / (max(cam_i) - min(cam_i))
            templates[:, i] = cam_i

    # X-axis in ms units
    templates_ts = np.linspace(-before * 1000, after * 1000, templates.shape[0], endpoint=False)

    fig, ax = plt.subplots(1, 1)
    mean_cam = np.mean(templates, axis=1)
    # mean_cam = savgol_filter(mean_cam, 11, 2)

    ax.plot(templates_ts, mean_cam, '-k', label='Activation')
    ax.set_xlabel('Time [ms]', fontdict={'size': 16})
    ax.set_ylabel('Amplitude [ms]', fontdict={'size': 16})
    plt.axvline(0, color='red', linestyle='--')

    if task == 'all':
        target_class = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
    elif task == 'wake_rem':
        target_class = {0: 'Wake', 1: 'REM'}
    elif task == 'rem_nrem':
        target_class = {0: 'NREM', 1: 'REM'}

    template_name_hloc = {'spindle': -2400, 'sw': -1000, 'emg': -500, 'rem': -2000}
    template_name_print = {'spindle': 'Spindle Start', 'sw': 'SW Start', 'emg': 'Movement Onset',
                           'rem': 'Eye movement '}

    ax.text(template_name_hloc[template_opt], text_height, template_name_print[template_opt],
            fontdict={'size': 12, 'style': 'italic', 'color': 'red'})
    plt.grid()
    plt.legend()
    plt.show()
    print(f'Averaged over {num_patients} patients and {num_templates} detections')
    return fig

def _get_templates(waveform, events, before, after, fs, idx1=0, idx2=5400, noise_lim_sec=0):

    # This is from class_activation_map_template.py

    # convert delimiters to samples
    before = int(before * fs)
    after = int(after * fs)
    noise_lim_samples = noise_lim_sec * fs  # no need to convert to int here, this will be done later

    # Sort events in ascending order
    events = np.sort(events)

    # Get number of sample points in waveform
    length = len(waveform)

    # Create empty list for templates
    templates = []

    # Create empty list for new rpeaks that match templates dimension
    rpeaks_new = np.empty(0, dtype=int)

    # Loop through R-Peaks
    events += np.random.uniform(low=-noise_lim_samples, high=noise_lim_samples, size=events.shape).astype(int)
    for event in events:

        # Before R-Peak
        a = event - before
        if a < 0:
            continue

        # After R-Peak
        b = event + after
        if b - idx1 > length:
            break

        if (event > idx1) & (event < idx2):
            # Append template list
            templates.append(waveform[a:b])

            # Append new rpeaks list
            events_new = np.append(rpeaks_new, events)

    # Convert list to numpy array
    templates = np.array(templates).T

    return templates, events_new


# next - 1slice_DS_frequency_lambda600_ls_remNrem_batch
#   1slice_DS_frequency_lambda20*0.25_lsb_remNrem_20_73
if __name__ == "__main__":
    file_name = "1slice_DS_Baseline_n_rem"
    extract_results(file_name)
    normalize_signals = False
    cuda_id = 2
    random = True
    oversample = False
    gap_norm_opt = 'None'
    task = 'rem_nrem'  # ['rem_nrem', 'wake_rem', 'all']
    num_patients = 1000  # There are 1570 patients in the test set
    cam_target = 0

    template_opt = 'spindle'  # in ['spindle', 'activation', 'sw', 'rem', 'emg']
    num_ch = 2
    before, after = 5, 5  # sec
    # 0.1 # 0.2 # 0.5 # 0.8 # 1
    noise_inter = 0.8
    templates = get_templates(file_name=file_name, template_opt=template_opt, cam_target=cam_target,
                              num_patients=num_patients, num_ch=num_ch, before=before, after=after, task=task,
                              try2load=False, cuda_id=cuda_id, random=random, gap_norm_opt=gap_norm_opt,
                              normalize_signals=normalize_signals, oversample=oversample, noise_inter=0)
    # plot_templates(file_name=file_name, cam_target=cam_target, text_height=0.3005, template_opt=template_opt,
    #                before=5, after=5, task=task, random=random)
