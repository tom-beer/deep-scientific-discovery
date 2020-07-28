import os
import numpy as np
import torch
import torch.utils.data
import pickle as pkl
from torch import nn, optim
from sklearn.metrics import f1_score, precision_recall_fscore_support

from networks import HSICClassifier, MLP1Layer
from ECG.train.datasets import create_kfoldcv_loaders
from ECG.train.train_utils import get_device

# experiment parameters
exp_name = 'main_task_lambda500_rr'
feature_subset = 'rr'

# training parameters
torch.manual_seed(42)
cuda_id = 0
batch_size = 32
lr = 0.0005
num_epochs = 40
rep_size = 512

file_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),  'saved_models', exp_name)

feature_opt = 'None' if 'baseline' in exp_name.lower() else 'HSIC+Concat'

device = get_device(cuda_id)

if not os.path.exists(file_dir):
    os.mkdir(file_dir)

feature_subset_dataloader = feature_subset  # if feature_subset == 'all_fcnet' else 'all'

kcv_loaders = create_kfoldcv_loaders(batch_size=batch_size, feature_subset=feature_subset_dataloader,
                                     feature_opt='HSIC+Concat', naf=True)

main_model = HSICClassifier(num_classes=2, feature_len=kcv_loaders[0][0].dataset.feature_len,
                            feature_opt=feature_opt, gap_norm_opt='batch_norm', in_channels=1).to(device)

main_model.load_state_dict(torch.load(os.path.join(file_dir, f'{exp_name}_params.pkl'), map_location='cpu'))
main_model.eval()


def train_rep2label(epoch, optimizer, model, criterion):
    model.train()
    train_loss = 0
    for batch_idx, (signal, labels, features, _, signal_names, _) in enumerate(train_loader):
        signal, labels, features = signal.to(device), labels.to(device), features.to(device)
        optimizer.zero_grad()
        _, _, gap = main_model(x=signal, features=features)
        logits = model(gap)
        loss = criterion(logits, labels)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset)}')
    return


def val_rep2label(best_acc, model, criterion):
    model.eval()
    correct, val_loss = 0, 0
    pred_list, label_list = [], []
    with torch.no_grad():
        for batch_idx, (signal, labels, features, _, signal_names, _) in enumerate(val_loader):
            signal, labels, features = signal.to(device), labels.to(device), features.to(device)
            _, _, gap = main_model(x=signal, features=features)
            logits = model(gap)
            loss = criterion(logits, labels)
            val_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)

            if torch.cuda.is_available():
                correct += (predicted.detach().cpu().numpy() == labels.detach().cpu().numpy()).sum()
            else:
                correct += (predicted == labels).sum()

            pred_list.append(predicted.detach().cpu().numpy())
            label_list.append(labels.detach().cpu().numpy())

        preds = np.concatenate(pred_list, axis=0)
        labels = np.concatenate(label_list, axis=0)

        f1_total = f1_score(labels, preds, labels=[0, 1, 2], average=None)[0:3]
        sum_samples = preds.shape[0]

        epoch_accuracy = 100 * float(correct) / sum_samples

        if epoch_accuracy >= best_acc:
            best_acc = epoch_accuracy
            torch.save(model.state_dict(), os.path.join(file_dir, f'{exp_name}_rep2label_params.pkl'))
            print(['Saved @  ' + str(epoch_accuracy) + '%'])

    val_loss /= len(val_loader)

    print('====> Val set Accuracy: {:.5f}'.format(epoch_accuracy))
    print(f'====> Val set F1: {f1_total[0]}, {f1_total[1]}, {f1_total[2]}')
    return best_acc


def test_rep2label(model, criterion):
    model.load_state_dict(torch.load(os.path.join(file_dir, f'{exp_name}_rep2label_params.pkl'), map_location='cpu'))
    model.eval()
    correct, test_loss = 0, 0
    pred_list, label_list = [], []
    with torch.no_grad():
        for batch_idx, (signal, labels, features, _, signal_names, _) in enumerate(test_loader):
            signal, labels, features = signal.to(device), labels.to(device), features.to(device)
            _, _, gap = main_model(x=signal, features=features)
            logits = model(gap)
            loss = criterion(logits, labels)
            test_loss += loss.item()

            _, predicted = torch.max(logits.data, 1)

            if torch.cuda.is_available():
                correct += (predicted.detach().cpu().numpy() == labels.detach().cpu().numpy()).sum()
            else:
                correct += (predicted == labels).sum()

            pred_list.append(predicted.detach().cpu().numpy())
            label_list.append(labels.detach().cpu().numpy())

        preds = np.concatenate(pred_list, axis=0)
        labels = np.concatenate(label_list, axis=0)

        sum_samples = preds.shape[0]

        f1_total = f1_score(labels, preds, labels=[0, 1, 2], average=None)[0:3]
        epoch_accuracy = 100 * float(correct) / sum_samples

    test_loss /= len(test_loader)

    perf_dict = {}
    perf_dict['accuracy'] = epoch_accuracy
    perf_dict['f1_normal'] = f1_total[0]
    perf_dict['f1_af'] = f1_total[1]
    perf_dict['f1_other'] = f1_total[2]
    for i, l in enumerate(['normal', 'af', 'other']):
        perf_dict[f'precision_{l}'] = \
            precision_recall_fscore_support(labels, preds, labels=[0, 1, 2], average=None)[0][i]
        perf_dict[f'recall_{l}'] = precision_recall_fscore_support(labels, preds, labels=[0, 1, 2], average=None)[1][i]

    print('====> Test set Accuracy: {:.5f}'.format(epoch_accuracy))
    print(f'====> Test set F1: {f1_total[0]}, {f1_total[1]}, {f1_total[2]}')
    return perf_dict


rep2label_perf = {}
val_acc_rep2label = []
for i_fold in range(5):
    train_loader, val_loader, test_loader = kcv_loaders[i_fold]

    rep2label_model = MLP1Layer(in_size=rep_size, hidden_size=rep_size, out_size=2).to(device)

    optimizer = optim.Adam(rep2label_model.parameters(), lr=lr, weight_decay=1e-6)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(num_epochs):
        train_rep2label(epoch, optimizer, rep2label_model, criterion)
        best_acc = val_rep2label(best_acc, rep2label_model, criterion)

    rep2label_perf[i_fold] = test_rep2label(rep2label_model, criterion)
    val_acc_rep2label.append(best_acc)


with open(os.path.join(file_dir, f'{exp_name}_rep2label_perf_ln.pkl'), 'wb') as handle:
    pkl.dump(rep2label_perf, handle, protocol=pkl.HIGHEST_PROTOCOL)

with open(os.path.join(file_dir, f'{exp_name}_val_acc_rep2label_ln.pkl'), 'wb') as handle:
    pkl.dump(val_acc_rep2label, handle, protocol=pkl.HIGHEST_PROTOCOL)
