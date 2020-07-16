import os
import numpy as np
import torch
import torch.utils.data
import pickle as pkl
from torch import nn, optim
from tqdm import tqdm
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from networks import MLP2Layer
from datasets import init_gap_datasets
from utils.train_utils import get_device, generate_gap, calc_save_perf_metrics, run_params
from features import feature_names_len_from_subset
from utils.rep_to_features import train_test_predict_features


def train_rep2label_nogap(gap_test, gap_train_loader, rep2label_model):
    rep2label_model.train()
    criterion = nn.CrossEntropyLoss()
    train_loss = 0
    for param_group in optimizer.param_groups:
        print(param_group["lr"])

    for batch_idx, (signal_names, labels) in tqdm(enumerate(gap_train_loader)):

        labels = labels.to(device)
        optimizer.zero_grad()
        gap = torch.stack([torch.from_numpy(gap_test[name]).to(device) for name in signal_names])
        gap = gap[:, :rep_size]
        logits = rep2label_model(gap)
        loss = criterion(logits, labels)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    scheduler.step()
    return


def val_test_rep2label_nogap(rep2label_model, loader, mode, gap_test, file_name, best_acc=None):
    assert mode in ['val', 'test']
    if mode == 'test':
        rep2label_model.load_state_dict(
            torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_models', file_name, f'{file_name}_rep2label_params.pkl'), map_location='cpu'))
    rep2label_model.eval()
    correct = 0
    pred_list, label_list = [], []
    with torch.no_grad():
        for batch_idx, (signal_names, labels) in tqdm(enumerate(loader)):
            labels = labels.to(device)
            gap = torch.stack([torch.from_numpy(gap_test[name]).to(device) for name in signal_names])
            gap = gap[:, :rep_size]
            logits = rep2label_model(gap)
            _, predicted = torch.max(logits.data, 1)

            if torch.cuda.is_available():
                correct += (predicted.detach().cpu().numpy() == labels.detach().cpu().numpy()).sum()
            else:
                correct += (predicted == labels).sum()

            pred_list.append(predicted.detach().cpu().numpy())
            label_list.append(labels.detach().cpu().numpy())

        preds = np.concatenate(pred_list, axis=0)
        labels = np.concatenate(label_list, axis=0)

        epoch_accuracy = 100 * float(correct) / loader.dataset.__len__()

        if mode == 'val' and epoch_accuracy >= best_acc:
            best_acc = epoch_accuracy
            torch.save(rep2label_model.state_dict(), os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_models', file_name, f'{file_name}_rep2label_params.pkl'))
            print(['Saved @  ' + str(epoch_accuracy) + '%'])

    results_dict = calc_save_perf_metrics(labels, preds, epoch_accuracy, mode=mode, file_name=file_name, save=False)
    if mode == 'val':
        return best_acc
    else:
        return results_dict


file_name = "1slice_DS_frequency_lambda20*0.25_lsb_remNrem_20_73"
normalize_signals = True
cuda_id = 1
feature_opt = 'HSIC+Concat'  # 'HSIC+Concat'
features_subset = ['frequency']  # ['frequency', 'spindle']
task = 'rem_nrem'  # ['rem_nrem', 'wake_rem', 'all']
oversample_gap = False  # False
gap_norm_opt = 'batch_norm'

num_epochs_rep2label = 40
lr_rep2label = 0.001
num_ch = 2
batch_size = 32
_, signal_len, one_slice, dataset_dir = run_params(file_name, features_subset, def_feature_opt='HSIC+Concat', task=task)
num_classes = 2 if 'rem' in task.lower() else 5
torch.manual_seed(44)
num_patients = 1e8

device = get_device(cuda_id)
file_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_models', file_name)

gap_train, gap_val, gap_test, rep_size = generate_gap(file_name, device, num_patients, task, batch_size,
                                                      normalize_signals, features_subset, gap_norm_opt)

rep2label_perf_list = []
for idx in range(5):
    gap_train_loader, gap_val_loader, gap_test_loader = init_gap_datasets(idx=idx, oversample=oversample_gap,
                                                                          batch_size=batch_size, modes=['train', 'val', 'test'])

    rep2label_model = MLP2Layer(in_size=rep_size, hidden_size1=256, hidden_size2=32, out_size=num_classes).to(device)

    optimizer = optim.Adam(rep2label_model.parameters(), lr=lr_rep2label, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, eta_min=lr_rep2label / 100, T_max=num_epochs_rep2label)

    best_acc = 0
    for epoch in range(1, num_epochs_rep2label + 1):
        train_rep2label_nogap(rep2label_model=rep2label_model, gap_test=gap_test,gap_train_loader=gap_train_loader)
        best_acc = val_test_rep2label_nogap(rep2label_model=rep2label_model, loader=gap_val_loader,
                                            mode='val', file_name=file_name, best_acc=best_acc, gap_test=gap_test)
        scheduler.step()

    rep2label_perf_list.append(val_test_rep2label_nogap(rep2label_model=rep2label_model,
                                          loader=gap_test_loader, mode='test',
                                          file_name=file_name, best_acc=best_acc,gap_test=gap_test))

with open(os.path.join(file_dir, f'{file_name}_rep2label_perf.pkl'), 'wb') as handle:
    pkl.dump(rep2label_perf_list, handle, protocol=pkl.HIGHEST_PROTOCOL)
