from torch import device
from sklearn.metrics import f1_score, cohen_kappa_score
import glob
import os
import numpy as np
import torch.utils.data
import pickle as pkl
from tqdm import tqdm
from networks import HSICClassifier
from EEG.datasets import init_datasets
import torch


def get_device(cuda_id):
    if not isinstance(cuda_id, str):
        cuda_id = str(cuda_id)
    is_cuda = True
    if 'data' in os.getcwd():  # if running from desktop
        is_cuda = False
    device_ = device("cuda:" + cuda_id if is_cuda else "cpu")
    return device_


def save_test_perf(results_dict, file_name):
    save_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_models', file_name, f'{file_name}_test_perf.pkl')
    with open(save_name, 'wb') as f:
        pkl.dump(results_dict, f, protocol=pkl.HIGHEST_PROTOCOL)
    return


def calc_save_perf_metrics(labels, preds, accuracy, mode, file_name, save=True, epoch=None):
    acc2 = 100*np.mean(labels == preds)
    if not np.isclose(accuracy, acc2):
        print("Warning: error in calc_save_perf_metrics")
    kappa_score = cohen_kappa_score(labels, preds)
    f1m = f1_score(labels, preds, average='micro')
    f1M = f1_score(labels, preds, average='macro')
    f1all = f1_score(labels, preds, average=None)
    num_classes = f1all.shape[0]
    mode_string = 'Validation' if mode == 'val' else 'Test'
    print(f'Epoch {epoch}: {mode_string} Performance -------------')
    print(f'Accuracy: {accuracy}')
    print(f'kappa_test: {kappa_score}, f1m: {f1m}, f1M: {f1M}')
    for i in range(num_classes):
        print(f'F{i}: {f1all[i]}', end=" ")
    print(" ")
    results_dict = {'accuracy': accuracy, 'kappa_score': kappa_score, 'f1m': f1m, 'f1M': f1M, 'f1all': f1all}
    if mode == 'test' and save:
        save_test_perf(results_dict, file_name)
    return results_dict


def update_train_stats(train_stats, avg_lambda, batch_size, lr, epoch_loss, hsic_loss, pval_list, num_batches,
                       total_feature_hsic):
    train_stats['lambda'].append(avg_lambda)
    train_stats['batch_size'].append(batch_size)
    train_stats['lr'].append(lr)
    train_stats['train_loss'].append(epoch_loss/num_batches)
    train_stats['hsic_loss'].append(hsic_loss/num_batches)  # hsic loss (or 0 for baseline models)
    train_stats['pval'].append(pval_list)
    train_stats['feature_hsic'].append(total_feature_hsic/num_batches)
    return train_stats


def unify_gap(mode, file_dir, file_name):
    gap_unified = {}
    for file in os.listdir(file_dir):
        if file.startswith(f'{file_name}_gap_{mode}_'):
            file2read = os.path.join(file_dir, file)
            with open(file2read, 'rb') as h:
                gap_dict = pkl.load(h)
            gap_unified.update(gap_dict)
            os.remove(file2read)

    for key, value in gap_unified.items():
        gap_unified[key] = value.cpu().detach().numpy()

    with open(os.path.join(file_dir, f'{file_name}_ugap_{mode}.pkl'), 'wb') as h:
        pkl.dump(gap_unified, h, protocol=pkl.HIGHEST_PROTOCOL)

    return gap_unified


def extract_gap(mode, loader, model, file_dir, file_name, device):
    gap_dict = {}

    for batch_idx, (signals, labels, sig_names, features) in enumerate(tqdm(loader)):
        data, features_rep = signals.to(device), features.to(device)
        _, _, gap = model(data, features_rep)

        gap = gap[:, :model.activation_size]
        update_gap_dict(sig_names, gap, gap_dict)

        with open(os.path.join(file_dir, f'{file_name}_gap_{mode}_{batch_idx}.pkl'), 'wb') as handle:
            pkl.dump(gap_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

        gap_dict = {}


def update_gap_dict(sig_names, gap, gap_dict):
    for i in range(len(sig_names)):
        gap_dict[sig_names[i]] = gap[i]


def generate_gap(file_name, device, num_patients, task, batch_size, normalize_signals, features_subset, gap_norm_opt):
    file_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_models', file_name)
    file_path = os.path.join(file_dir, f'{file_name}_ugap_test.pkl')
    if not os.path.exists(file_path):
        num_classes = 2 if 'rem' in task.lower() else 5
        feature_opt, signal_len, one_slice, dataset_dir = run_params(file_name, features_subset,
                                                                     def_feature_opt='HSIC+Concat', task=task)

        train_loader, val_loader, test_loader = init_datasets(num_patients=num_patients, dataset_dir=dataset_dir,
                                                              task=task, one_slice=True, batch_size=batch_size, conv_d=1,
                                                              oversample=False, normalize_signals=normalize_signals,
                                                              features_subset=features_subset, num_ch=2,
                                                              low_sp=False, modes=['train', 'val', 'test'])

        haifa_model = HSICClassifier(num_classes=num_classes, signal_len=signal_len, feature_opt=feature_opt,
                                     gap_norm_opt=gap_norm_opt,
                                     feature_len=train_loader.dataset.feature_len, in_channels=2).to(device)

        haifa_model.load_state_dict(torch.load(os.path.join(file_dir, f'{file_name}_params.pkl'), map_location='cpu'))

        gap_train, gap_val, gap_test, rep_size = \
            generate_gap_internal(train_loader, val_loader, test_loader, haifa_model, file_dir, file_name, device)
    else:
        with open(os.path.join(file_dir, f'{file_name}_ugap_train.pkl'), 'rb') as h:
            gap_train = pkl.load(h)

        with open(os.path.join(file_dir, f'{file_name}_ugap_val.pkl'), 'rb') as h:
            gap_val = pkl.load(h)

        with open(os.path.join(file_dir, f'{file_name}_ugap_test.pkl'), 'rb') as h:
            gap_test = pkl.load(h)

    for key, value in gap_test.items():
        rep_size = value.shape[0]
        break

    return gap_train, gap_val, gap_test, rep_size


def generate_gap_internal(train_loader, val_loader, test_loader, model, file_dir, file_name, device):

    file_path = os.path.join(file_dir, f'{file_name}_ugap_test.pkl')
    if not os.path.exists(file_path):
        extract_gap('train', train_loader, model=model, file_dir=file_dir, file_name=file_name, device=device)
        extract_gap('val', val_loader, model=model, file_dir=file_dir, file_name=file_name, device=device)
        extract_gap('test', test_loader, model=model, file_dir=file_dir, file_name=file_name, device=device)
        print('extraction done')

        gap_train = unify_gap('train', file_dir=file_dir, file_name=file_name)
        gap_val = unify_gap('val', file_dir=file_dir, file_name=file_name)
        gap_test = unify_gap('test', file_dir=file_dir, file_name=file_name)

        for f in glob.glob(f'{file_dir}/{file_name}_gap*'):
            os.remove(f)
    else:
        with open(os.path.join(file_dir, f'{file_name}_ugap_train.pkl'), 'rb') as h:
            gap_train = pkl.load(h)

        with open(os.path.join(file_dir, f'{file_name}_ugap_val.pkl'), 'rb') as h:
            gap_val = pkl.load(h)

        with open(os.path.join(file_dir, f'{file_name}_ugap_test.pkl'), 'rb') as h:
            gap_test = pkl.load(h)

    for key, value in gap_test.items():
        rep_size = value.shape[0]
        break

    return gap_train, gap_val, gap_test, rep_size


def col_names_to_idx(subset, df):
    return[df.columns.get_loc(col) for col in subset]


def run_params(file_name, features_subset, def_feature_opt='HSIC+Concat', task='all'):
    assert def_feature_opt in ['HSIC', 'HSIC+Concat']
    assert task in ['all', 'wake_rem', 'rem_nrem']
    if features_subset:
        feature_opt = def_feature_opt
    else:
        feature_opt = 'None'
    signal_len = 9600 if 'ds' in file_name.lower() else 15000
    signal_len = signal_len / 4 if 'slice' in file_name.lower() else signal_len
    one_slice = 'slice' in file_name.lower()
    dataset_dir = 'filtered0.05' if 'ds' in file_name.lower() else 'nofilter0.05'
    if 'rem' in task:
        dataset_dir = 'ALL0.05'
    print(f'dataset_dir = {dataset_dir}')
    return feature_opt, signal_len, one_slice, dataset_dir
