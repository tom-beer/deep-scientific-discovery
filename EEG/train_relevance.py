import os
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from tqdm import tqdm

from networks import MLP1Layer
from EEG.datasets import init_datasets
from EEG.feature_utils import feature_names_len_from_subset
from EEG.train_utils import get_device, calc_save_perf_metrics

# experiment parameters
file_name = 'dev'
lambda_hsic = 600
features_subset = ['frequency']
task = 'rem_nrem'
balanced_dataset = False

# training parameters
cuda_id = 0
num_epochs = 40
batch_size = 32
lr = 0.00003

torch.manual_seed(44)
device = get_device(cuda_id)
file_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_models', file_name)
if not os.path.exists(file_dir):
    os.mkdir(file_dir)

print(f'{file_name} Started training')
train_loader, val_loader, test_loader = init_datasets(task=task, balanced_dataset=True, normalize_signals=False,
                                                      batch_size=batch_size, features_subset=features_subset)
feature_len = feature_names_len_from_subset(features_subset)[0]
model = MLP1Layer(in_size=feature_len, hidden_size=128, out_size=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
criterion = nn.CrossEntropyLoss()


def train():
    model.train()
    train_loss = 0
    for batch_idx, (_, labels, _, features) in enumerate(tqdm(train_loader)):
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(features)
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
            logits = model(features)
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
