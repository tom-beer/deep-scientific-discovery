import os
import numpy as np
import pickle as pkl
import torch
import torch.utils.data
from torch import nn, optim
from tqdm import tqdm
from sklearn.metrics import f1_score

from ECG.train.learning_rate_schedulers import AnnealingRestartScheduler
from ECG.train.datasets import create_dataloaders
from hsic import HSICLoss
from ECG.train.train_utils import get_device
from networks import HaifaNetVPT


lambda_hsic = 500
cuda_id = str(2)

file_name = f"lambda_{lambda_hsic}_rr_naf_hsic_nocat"
torch.manual_seed(42)
feature_opt = 'HSIC+Concat'  # 'HSIC+Concat'  # {'None', 'Concat', 'HSIC', 'HSIC+Concat'}
feature_subset = 'rr'
gap_norm_opt = 'batch_norm'

naf = True

update_batch_size = False
update_lambda = True

init_sigma_gap = 1
lr = 0.001
num_epochs = 70
epoch2save = 0
in_channels_ = 1
num_classes = 3
log_interval = 1079

batch_size = 48

if update_batch_size:
    batch_size = 16

file_dir = os.path.join('saved_models', file_name)
if not os.path.exists(file_dir):
    os.mkdir(file_dir)

device = get_device(cuda_id)
base_dir = ".."

train_loader, val_loader, _ = create_dataloaders(batch_size, feature_subset, feature_opt, naf)

model = HaifaNetVPT(num_classes=num_classes, feature_len=train_loader.dataset.feature_len, feature_opt=feature_opt,
                    gap_norm_opt=gap_norm_opt).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.95, 0.99), eps=1e-08, weight_decay=0, amsgrad=False)
classification_criterion = nn.CrossEntropyLoss()
independence_criterion = HSICLoss(feature_opt, lambda_hsic, model.activation_size, device, decay_factor=0.7,
                                  external_feature_std=3)
lr_scheduler = AnnealingRestartScheduler(lr_min=lr/100, lr_max=lr, steps_per_epoch=len(train_loader),
                                         lr_max_decay=0.6, epochs_per_cycle=num_epochs, cycle_length_factor=1.5)
lambda_vec = lambda_hsic * np.hstack([np.linspace(0, 1, num_epochs//2), np.ones(num_epochs-num_epochs//2)])


def train(train_loader, epoch, sigma_gap, lambda_hsic):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target, feature, _, _, feature_rep) in enumerate(tqdm(train_loader)):
        data, target, feature, feature_rep = data.to(device), target.to(device), feature.to(device), feature_rep.to(device)
        optimizer.zero_grad()
        for g in optimizer.param_groups:
            g['lr'] = lr_scheduler.lr
        logits, _, gap = model(data, feature_rep)

        classification_loss = classification_criterion(logits, target)
        hsic_loss = independence_criterion.calc_loss(gap, feature)

        loss = classification_loss + hsic_loss

        loss.backward()
        train_loss += loss.item()

        _, predicted = torch.max(logits.data, 1)

        if torch.cuda.is_available():
            correct += (predicted.detach().cpu().numpy() == target.detach().cpu().numpy()).sum()
        else:
            correct += (predicted == target).sum()

        optimizer.step()

        if batch_idx % log_interval == 0:

            loss_total = loss.item() / len(data)
            print('\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                loss_total))

        lr_scheduler.on_batch_end_update()
        if update_batch_size:
            if epoch == 20:
                train_loader = torch.utils.data.DataLoader(dataset=train_loader.dataset, batch_size=32, shuffle=False,
                                                           sampler=train_loader.dataset.sampler)
            if epoch == 40:
                train_loader = torch.utils.data.DataLoader(dataset=train_loader.dataset, batch_size=48, shuffle=False,
                                                           sampler=train_loader.dataset.sampler)
        if update_lambda:
            lambda_hsic = lambda_vec[epoch-1]

    epoch_accuracy = 100 * float(correct) / train_loader.dataset.__len__()
    print(f'Training Accuracy: {epoch_accuracy }')
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))
    print(f'Sigma = {sigma_gap}')

    return train_loader, sigma_gap, lambda_hsic


def validation(perf_dict, epoch):
    model.eval()
    val_loss = 0
    correct = 0
    pred_list = []
    label_list = []
    with torch.no_grad():
        for batch_idx, (data, target, feature, _, _, feature_rep) in enumerate(val_loader):
            data, target, feature, feature_rep = data.to(device), target.to(device), feature.to(device), feature_rep.to(device)
            logits, _, gap = model(data, feature_rep)

            hsic_loss = independence_criterion.calc_loss(gap, feature)

            loss = classification_criterion(logits, target) + hsic_loss

            val_loss += loss.item()

            _, predicted = torch.max(logits.data, 1)

            if torch.cuda.is_available():
                correct += (predicted.detach().cpu().numpy() == target.detach().cpu().numpy()).sum()
            else:
                correct += (predicted == target).sum()

            pred_list.append(predicted.detach().cpu().numpy())
            label_list.append(target.detach().cpu().numpy())

    preds = np.concatenate(pred_list, axis=0)
    labels = np.concatenate(label_list, axis=0)

    f1_total = f1_score(labels, preds, labels=[0, 1, 2], average=None)[0:3]

    val_loss /= len(val_loader.dataset)
    epoch_accuracy = 100 * float(correct) / val_loader.dataset.__len__()

    if (epoch > epoch2save):
        perf_dict['accuracy'].append(epoch_accuracy)
        perf_dict['f1']['normal'].append(f1_total[0])
        perf_dict['f1']['af'].append(f1_total[1])
        perf_dict['f1']['other'].append(f1_total[2])

        if (epoch_accuracy >= np.max(perf_dict['accuracy'])):
            torch.save(model.state_dict(), os.path.join(file_dir, f'{file_name}_params.pkl'))
            print(['Saved @  ' + str(epoch_accuracy) + '%'])

    print('====> Validation set loss: {:.5f}'.format(val_loss))
    print('Validation accuracy: {:.4f}'.format(epoch_accuracy))
    print(f'Validation F1: {f1_total[0]}, {f1_total[1]}, {(f1_total[2])}')

    with open(os.path.join(file_dir, f'{file_name}_perf_dict.pkl'), 'wb') as handle:
        pkl.dump(perf_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

    return perf_dict


if __name__ == "__main__":
    perf_dict = {'accuracy': [], 'f1': {'normal': [], 'af': [], 'other': []}}
    print(file_name)
    sigma_gap = init_sigma_gap
    for epoch in range(1, num_epochs + 1):
        train_loader, sigma_gap, lambda_hsic = train(train_loader, epoch, sigma_gap, lambda_hsic)
        perf_dict = validation(perf_dict, epoch)
        lr_scheduler.on_epoch_end_update(epoch=epoch)
    print(f'{file_name} finished training')
