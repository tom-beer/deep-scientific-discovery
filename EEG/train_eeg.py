import os
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
import pickle as pkl
from sklearn.metrics import pairwise_distances as pdist
from collections import defaultdict
from networks import HaifaNetVPT
from EEG.datasets import init_datasets
from hsic import HSIC
from EEG.train_utils import get_device, calc_save_perf_metrics, update_train_stats, run_params


lambda_hsic = 600
features_subset = ['frequency']  # ['spindle', 'frequency']
cuda_id = 3
oversample = False
normalize_signals = True
update_lambda = True
update_batch_size = False
need_train = True
use_lr_sched = True
warm_start = 20
lambda_prop = 0.25
task = 'rem_nrem'  # ['rem_nrem', 'wake_rem', 'all']


gap_norm_opt = 'None' #'batch_norm'
num_ch = 2
low_sp = False
random_flag = False
num_epochs = 60
batch_size = 32
lr = 0.00003
num_patients = 1e8
lsb = ''
if update_lambda:
    lsb += 'l'
if use_lr_sched:
    lsb += 's'
if update_batch_size:
    lsb += 'b'
ch = '_1ch' if num_ch == 1 else ''
low = '_low' if low_sp else ''

wrem = '_WREM' if task == 'wake_rem' else '_remNrem' if task == 'rem_nrem' else ''
balanced = '_balanced' if oversample else ''

feature_names = '_'
for item in features_subset:
    feature_names += item + '_'

# file_name = '1slice_DS_Baseline_balanced'
file_name = 'test'#f"1slice_DS{feature_names}lambda{lambda_hsic}*{lambda_prop}_{lsb}{ch}{low}{wrem}_{warm_start}{balanced}_include_wake"
hsic_kernel = 'Linear' if 'linear' in file_name.lower() else 'Gaussian'
num_classes = 2 if 'rem' in task.lower() else 5
feature_opt, signal_len, one_slice, dataset_dir = run_params(file_name, features_subset, def_feature_opt='HSIC+Concat',
                                                             task=task)
torch.manual_seed(44)

device = get_device(cuda_id)

file_dir = os.path.join('saved_models', file_name)
if not os.path.exists(file_dir):
    os.mkdir(file_dir)

if warm_start < num_epochs:
    lambda_vec = lambda_hsic * np.hstack([np.linspace(0, 1, warm_start), np.ones(100)])
else:
    lambda_vec = lambda_hsic * np.hstack([np.linspace(0, 1, num_epochs//6), np.ones(num_epochs-num_epochs//6)])

print(f'{file_name} started training')
train_loader, val_loader, test_loader = init_datasets(num_patients=num_patients, dataset_dir=dataset_dir, num_ch=num_ch,
                                                      one_slice=one_slice, batch_size=batch_size, conv_d=1, low_sp=low_sp,
                                                      features_subset=features_subset, random_flag=random_flag,
                                                      task=task, oversample=oversample, normalize_signals=normalize_signals)
print(f'Median Distance of Features: {train_loader.dataset.med_dist}')
model = HaifaNetVPT(num_classes=num_classes, signal_len=signal_len, feature_opt=feature_opt,
                    feature_len=train_loader.dataset.feature_len, in_channels=num_ch, gap_norm_opt=gap_norm_opt).to(device)
print(f'Feature len: {train_loader.dataset.feature_len}')
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
if use_lr_sched:
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, eta_min=lr/100, T_max=num_epochs)
criterion = nn.CrossEntropyLoss()


def train(train_loader, sigma_gap, train_stats):
    if update_lambda:
        curr_lambda = lambda_vec[epoch - 1]
    else:
        curr_lambda = lambda_hsic
    model.train()
    epoch_loss = 0
    epoch_classification_loss = 0
    total_feature_hsic = 0

    pval_list = []
    count_0 = []
    lambdas_list = []
    num_batches = len(train_loader)
    for batch_idx, (signals, labels, _, features) in enumerate(tqdm(train_loader)):
        signals, labels, features = signals.to(device), labels.to(device), features.to(device)

        optimizer.zero_grad()
        preds, _, gap = model(signals, features)
        count_0.append((labels == 0).cpu().numpy().mean())
        classification_loss = criterion(preds, labels)
        hsic_features = 0

        if 'hsic' in feature_opt.lower():
            med_dist_gap = np.median(pdist(gap.detach().cpu().numpy(), metric='euclidean').reshape(-1, 1))
            sigma_gap = np.maximum(0.7 * sigma_gap + 0.3 * med_dist_gap, 0.005)
            gap = gap[:, :model.activation_size]
            hsic_features = HSIC(gap, features, kernelX=hsic_kernel, kernelY=hsic_kernel,
                                 sigmaX=sigma_gap, sigmaY=train_loader.dataset.med_dist, device=device)

            # real_sigma_gap = np.maximum(med_dist_gap, 0.005)
            # hsic_features_real_gap = HSIC(gap, features, kernelX=hsic_kernel, kernelY=hsic_kesrnel,
            #                      sigmaX=real_sigma_gap, sigmaY=train_loader.dataset.med_dist, device=device)

            # if not epoch % 10:  # this takes to much time..theses epochs take 1.5 hours :(
            #     pval_list += [ind_perm_test(x=gap, y=features, sigmaX=sigma_gap, sigmaY=train_loader.dataset.med_dist,
            #                   hsic_test=hsic_features, device=device, num_reps=1000, hsic_kernel=hsic_kernel)]

            if epoch > warm_start:
                curr_lambda = lambda_hsic
                # if hsic_features.item() > 0.0005:
                #     curr_lambda = lambda_prop * classification_loss.item() / hsic_features.item()
            total_feature_hsic += hsic_features.item()

            hsic_loss = curr_lambda * hsic_features

            loss = classification_loss + hsic_loss
        else:
            loss = classification_loss

        try:
            loss.backward()
        except:
            pass
        lambdas_list.append(curr_lambda)
        epoch_loss += loss.item()
        epoch_classification_loss += classification_loss.item()
        optimizer.step()
        if update_batch_size:
            if epoch == 10:
                train_loader = DataLoader(dataset=train_loader.dataset, batch_size=32, shuffle=True)
    print(f"Class 0 prop: {np.mean(count_0)}")

    if use_lr_sched:
        scheduler.step()
    for param_group in optimizer.param_groups:
        curr_lr = param_group["lr"]
        print(f'LR: {curr_lr}')
    avg_lambda = np.mean(lambda_hsic)
    print(f"avg_hsic_lambda: {avg_lambda}")
    train_stats = update_train_stats(train_stats, avg_lambda=avg_lambda, batch_size=batch_size, lr=curr_lr,
                                     epoch_loss=epoch_loss, hsic_loss=epoch_loss-epoch_classification_loss,
                                     pval_list=pval_list, num_batches=num_batches, total_feature_hsic=total_feature_hsic)
    return train_loader, sigma_gap, train_stats


def val_or_test(loader, mode='val', accuracy=None, file_name=None, epoch=None, train_stats=None):
    if mode == 'test':
        model.load_state_dict(torch.load(os.path.join(file_dir, f'{file_name}_params.pkl'), map_location='cpu'))
    model.eval()
    correct = 0
    pred_list = []
    label_list = []
    epoch_loss = 0
    with torch.no_grad():
        for batch_idx, (signals, labels, _, features) in enumerate(tqdm(loader)):
            signals, labels, features = signals.to(device), labels.to(device), features.to(device)
            preds, _, gap = model(signals, features)
            loss = criterion(preds, labels)
            _, predicted = torch.max(preds.data, 1)

            if torch.cuda.is_available():
                correct += (predicted.detach().cpu().numpy() == labels.detach().cpu().numpy()).sum()
            else:
                correct += (predicted == labels).sum()

            pred_list.append(predicted.detach().cpu().numpy())
            label_list.append(labels.detach().cpu().numpy())
            epoch_loss += loss.item()

    preds = np.concatenate(pred_list, axis=0)
    labels = np.concatenate(label_list, axis=0)

    epoch_accuracy = 100 * float(correct) / loader.dataset.__len__()
    calc_save_perf_metrics(labels=labels, preds=preds, accuracy=epoch_accuracy, epoch=epoch, mode=mode,
                           file_name=file_name)

    if mode == 'val':
        train_stats['val_loss'].append(epoch_loss/loader.__len__())
        train_stats['val_accuracy'].append(epoch_accuracy)
        if epoch_accuracy >= accuracy and ((epoch > num_epochs / 2) or (not update_lambda)):
            accuracy = epoch_accuracy
            torch.save(model.state_dict(), os.path.join(file_dir, f'{file_name}_params.pkl'))
            print(f'Saved @ {accuracy}')

    return accuracy, train_stats


if __name__ == "__main__":
    accuracy = 0
    sigma_gap = 1
    need_test = True
    train_stats = defaultdict(list)
    if need_train:
        try:
            for epoch in range(1, num_epochs + 1):
                train_loader, sigma_gap, train_stats = train(train_loader, sigma_gap, train_stats)
                accuracy, train_stats = val_or_test(loader=val_loader, train_stats=train_stats, mode='val',
                                                    accuracy=accuracy, file_name=file_name, epoch=epoch)
                with open(os.path.join(file_dir, f'{file_name}_train_stats.pkl'), 'wb') as handle:
                    pkl.dump(train_stats, handle, protocol=pkl.HIGHEST_PROTOCOL)

        except KeyboardInterrupt:

            val_or_test(loader=test_loader, mode='test', file_name=file_name)
            need_test = False
    if need_test:
        val_or_test(loader=test_loader, mode='test', file_name=file_name)
        with open(os.path.join(file_dir, f'{file_name}_train_stats.pkl'), 'wb') as handle:
            pkl.dump(train_stats, handle, protocol=pkl.HIGHEST_PROTOCOL)

    print(f'{file_name} finished training')
