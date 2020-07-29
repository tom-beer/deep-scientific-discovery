import os
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from tqdm import tqdm

from networks import HSICClassifier
from hsic import HSICLoss
from EEG.datasets import init_datasets
from EEG.train_utils import get_device, calc_save_perf_metrics, run_params

# experiment parameters
file_name = 'dev'
lambda_hsic = 600
features_subset = ['frequency']
task = 'rem_nrem'
balanced_dataset = False

# training parameters
cuda_id = 0
warm_start = 20
num_epochs = 100
batch_size = 32
lr = 0.00003

feature_opt = run_params(features_subset)

torch.manual_seed(44)
device = get_device(cuda_id)
file_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_models', file_name)
if not os.path.exists(file_dir):
    os.mkdir(file_dir)

lambda_vec = lambda_hsic * np.hstack([np.linspace(0, 1, warm_start), np.ones(100)])

print(f'{file_name} started training')
train_loader, val_loader, test_loader = init_datasets(batch_size=batch_size, features_subset=features_subset,
                                                      task=task, balanced_dataset=balanced_dataset,
                                                      normalize_signals=True)
model = HSICClassifier(num_classes=2, feature_opt=feature_opt, feature_len=train_loader.dataset.feature_len,
                       in_channels=2, gap_norm_opt='None').to(device)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, eta_min=lr/100, T_max=num_epochs)
classification_criterion = nn.CrossEntropyLoss()
independence_criterion = HSICLoss(feature_opt, lambda_hsic, model.activation_size, device, decay_factor=0.7,
                                  external_feature_std=train_loader.dataset.med_dist)


def train(epoch):
    model.train()
    epoch_loss = 0

    for batch_idx, (signals, labels, _, features) in enumerate(tqdm(train_loader)):
        signals, labels, features = signals.to(device), labels.to(device), features.to(device)

        optimizer.zero_grad()
        preds, _, gap = model(signals, features)
        classification_loss = classification_criterion(preds, labels)
        hsic_loss = independence_criterion.calc_loss(gap, features)
        loss = classification_loss + hsic_loss
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()

    scheduler.step(epoch=epoch)
    return


def val_or_test(loader, mode='val', accuracy=None, file_name=None, epoch=None):
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
            loss = classification_criterion(preds, labels)
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
        if epoch_accuracy >= accuracy and (epoch > num_epochs / 2):
            accuracy = epoch_accuracy
            torch.save(model.state_dict(), os.path.join(file_dir, f'{file_name}_params.pkl'))
            print(f'Saved @ {accuracy}')

    return accuracy


if __name__ == "__main__":
    accuracy = 0
    need_test = True
    try:
        for epoch in range(num_epochs):
            train(epoch=epoch)
            accuracy = val_or_test(loader=val_loader, mode='val', accuracy=accuracy,
                                   file_name=file_name, epoch=epoch)
    except KeyboardInterrupt:

        val_or_test(loader=test_loader, mode='test', file_name=file_name)
        need_test = False
    if need_test:
        val_or_test(loader=test_loader, mode='test', file_name=file_name)

    print(f'{file_name} finished training')
