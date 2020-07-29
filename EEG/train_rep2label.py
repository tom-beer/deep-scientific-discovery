import os
import numpy as np
import torch
import torch.utils.data
import pickle as pkl
from torch import nn, optim
from tqdm import tqdm
from networks import HSICClassifier, MLP2Layer
from EEG.datasets import create_kfoldcv_loaders
from EEG.train_utils import get_device, calc_save_perf_metrics, run_params

# Experiment parameters
file_name = "1slice_DS_frequency_lambda20*0.25_lsb_remNrem_20_73"
task = 'rem_nrem'
features_subset = ['frequency']  # ['frequency', 'spindle']
balanced_dataset = True

# Training parameters
cuda_id = 0
batch_size = 32
num_epochs = 40
lr = 0.0005
rep_size = 512

feature_opt = run_params(features_subset)
torch.manual_seed(44)
device = get_device(cuda_id)
file_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_models', file_name)

kcv_loaders = create_kfoldcv_loaders(task=task, balanced_dataset=balanced_dataset, normalize_signals=True,
                                     batch_size=32, features_subset=features_subset)

main_model = HSICClassifier(num_classes=2, feature_opt=feature_opt, gap_norm_opt='batch_norm',
                            feature_len=kcv_loaders[0][0].dataset.feature_len, in_channels=2).to(device)

main_model.load_state_dict(torch.load(os.path.join(file_dir, f'{file_name}_params.pkl'), map_location='cpu'))
main_model.eval()


def train_rep2label(optimizer, model, criterion, train_loader):
    model.train()
    train_loss = 0
    for batch_idx, (signals, labels, _, features) in enumerate(tqdm(train_loader)):
        signals, labels, features = signals.to(device), labels.to(device), features.to(device)
        _, _, gap = main_model(x=signals, features=features)

        optimizer.zero_grad()
        logits = model(gap)
        loss = criterion(logits, labels)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    scheduler.step()

    for param_group in optimizer.param_groups:
        print(param_group['lr'])
    return


def val_test_rep2label(rep2label_model, loader, mode, file_name, best_acc=None):
    assert mode in ['val', 'test']
    if mode == 'test':
        rep2label_model.load_state_dict(
            torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_models', file_name, f'{file_name}_rep2label_params.pkl'), map_location='cpu'))
    rep2label_model.eval()
    correct = 0
    pred_list, label_list = [], []
    with torch.no_grad():
        for batch_idx, (signals, labels, _, features) in enumerate(train_loader):
            signals, labels, features = signals.to(device), labels.to(device), features.to(device)
            _, _, gap = main_model(x=signals, features=features)

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


if __name__ == "__main__":

    rep2label_perf = {}
    val_acc_rep2label = []

    for i_fold in range(5):

        train_loader, val_loader, test_loader = kcv_loaders[i_fold]
        rep2label_model = MLP2Layer(in_size=rep_size, hidden_size1=256, hidden_size2=32, out_size=2).to(device)
        optimizer = optim.Adam(rep2label_model.parameters(), lr=lr, weight_decay=1e-6)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, eta_min=lr / 100, T_max=num_epochs)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0
        for epoch in range(num_epochs):
            train_rep2label(optimizer, rep2label_model, criterion=criterion, train_loader=train_loader)
            best_acc = val_test_rep2label(rep2label_model=rep2label_model, loader=val_loader, mode='val',
                                          file_name=file_name, best_acc=best_acc)
        rep2label_perf[i_fold] = val_test_rep2label(rep2label_model=rep2label_model, loader=test_loader, mode='test',
                                                    file_name=file_name, best_acc=best_acc)
        val_acc_rep2label.append(best_acc)

        with open(os.path.join(file_dir, f'{file_name}_rep2label_perf.pkl'), 'wb') as handle:
            pkl.dump(rep2label_perf, handle, protocol=pkl.HIGHEST_PROTOCOL)

        with open(os.path.join(file_dir, f'{file_name}_val_acc_rep2label.pkl'), 'wb') as handle:
            pkl.dump(val_acc_rep2label, handle, protocol=pkl.HIGHEST_PROTOCOL)
