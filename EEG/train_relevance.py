import os
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from tqdm import tqdm
from networks import FCFeatureNet
from EEG.datasets import init_datasets
from EEG.feature_utils import feature_names_len_from_subset
from EEG.train_utils import get_device, calc_save_perf_metrics

file_name = 'rem2remnrem' #"rem_nrem_balanced_frq"
features_subset = ['rem']
cuda_id = 2
dataset_dir = 'ALL0.05'
num_patients = 1e8
one_slice = True
task = 'rem_nrem'  # ['rem_nrem', 'wake_rem', 'all']
low_sp = False
oversample = True

num_classes = 2 if 'rem' in task.lower() else 5
num_epochs = 40
batch_size = 32
lr = 0.0003
torch.manual_seed(44)

device = get_device(cuda_id)

file_dir = os.path.join('saved_models', file_name)
if not os.path.exists(file_dir):
    os.mkdir(file_dir)

print(f'{file_name} Started training')
train_loader, val_loader, test_loader = init_datasets(num_patients=num_patients, dataset_dir=dataset_dir,
                                                      batch_size=batch_size, conv_d=1, features_subset=features_subset,
                                                      one_slice=one_slice, task=task, low_sp=low_sp,
                                                      oversample=oversample, normalize_signals=False)
feature_len = feature_names_len_from_subset(features_subset)[0]
model = FCFeatureNet(num_classes=num_classes, feature_len=feature_len).to(device)
# model = multi_FC_FeatureNet(rep_size=feature_len, out_size=num_classes)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
criterion = nn.CrossEntropyLoss()

print(f'num_patients - {num_patients}')


def train():
    model.train()
    train_loss = 0
    for batch_idx, (_, labels, _, features) in enumerate(tqdm(train_loader)):
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(features)[0]
        classification_loss = criterion(logits, labels)
        loss = classification_loss
        loss.backward()
        train_loss += loss.item()
        optimizer.step()


def val_or_test(loader, mode='val', accuracy=None, file_name=None, epoch=None):
    if mode == 'test':
        model.load_state_dict(torch.load(os.path.join(file_dir, f'{file_name}_params.pkl'), map_location='cpu'))
    model.eval()
    correct = 0
    pred_list = []
    label_list = []
    with torch.no_grad():
        for batch_idx, (_, labels, _, features) in enumerate(tqdm(loader)):
            features, labels = features.to(device), labels.to(device)
            logits = model(features)[0]
            _, predicted = torch.max(logits.data, 1)

            if torch.cuda.is_available():
                correct += (predicted.detach().cpu().numpy() == labels.detach().cpu().numpy()).sum()
            else:
                correct += (predicted == labels).sum()

            pred_list.append(predicted.detach().cpu().numpy())
            label_list.append(labels.detach().cpu().numpy())

    preds = np.concatenate(pred_list, axis=0)
    labels = np.concatenate(label_list, axis=0)

    # calculate kappa and f1-score
    epoch_accuracy = 100 * float(correct) / loader.dataset.__len__()

    calc_save_perf_metrics(labels, preds, epoch_accuracy, epoch, mode, file_name)

    if mode == 'val' and epoch_accuracy > accuracy:
        accuracy = epoch_accuracy
        torch.save(model.state_dict(), os.path.join(file_dir, f'{file_name}_params.pkl'))
        print(f'Saved @ {accuracy}')
    if mode == 'test':
        calc_save_perf_metrics(labels=labels, preds=preds, accuracy=epoch_accuracy, epoch=epoch, mode=mode,
                               file_name=file_name)

    return accuracy


if __name__ == "__main__":
    accuracy = 0
    need_test = True
    try:
        for epoch in range(1, num_epochs + 1):
            train()
            accuracy = val_or_test(loader=val_loader, mode='val', accuracy=accuracy, epoch=epoch, file_name=file_name)
    except KeyboardInterrupt:
        val_or_test(loader=test_loader, mode='test', file_name=file_name)
        need_test = False
    if need_test:
        val_or_test(loader=test_loader, mode='test', file_name=file_name)
    print(f'{file_name} finished training')
