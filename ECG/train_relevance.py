import os
import numpy as np
import torch
import torch.utils.data
import pickle as pkl
from torch import nn, optim
from tqdm import tqdm
from sklearn.metrics import f1_score

from networks import MLP1Layer
from ECG.train.datasets import create_dataloaders
from ECG.train.train_utils import get_device

# experiment parameters
feature_subset = 'rr'
exp_name = f"relevance_{feature_subset}"
learn_rep = True

# training parameters
torch.manual_seed(42)
cuda_id = 0
batch_size = 48
num_epochs = 40
lr = 0.005

# folders and names..
file_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_models', exp_name)
if not os.path.exists(file_dir):
    os.mkdir(file_dir)
extracted_rep_file_name = os.path.join(file_dir, f'{exp_name}_rep.pkl')
param_file_name = os.path.join(file_dir, f'{exp_name}_params.pkl')
perf_dict_name = os.path.join(file_dir, f'{exp_name}_perf.pkl')
print(exp_name)

device = get_device(cuda_id)

feature_opt = 'Concat'  # just so they will be available in ECGDataset

train_loader, val_loader, test_loader = create_dataloaders(batch_size, feature_subset, feature_opt, naf=True)

model = MLP1Layer(in_size=train_loader.dataset.feature_len, hidden_size=128, out_size=2).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
criterion = nn.CrossEntropyLoss()


def train(epoch):
    model.train()
    class_balance_mean_list = []
    train_loss = 0
    for batch_idx, (_, target, feature, _, _, _) in enumerate(tqdm(train_loader)):
        target, feature = target.to(device), feature.to(device)
        optimizer.zero_grad()
        logits = model(feature)[0]
        loss = criterion(logits, target)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        class_balance_mean_list.append(target.cpu().numpy().sum()/train_loader.batch_size)
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')
    print(f"Balance in training set is {np.mean(class_balance_mean_list):2.2f}")
    return


def val(accuracy):
    model.eval()
    correct = 0
    pred_list, label_list = [], []
    with torch.no_grad():
        for batch_idx, (_, target, feature, _, _, _) in enumerate(val_loader):
            target, feature = target.to(device), feature.to(device)
            logits = model(feature)[0]

            _, predicted = torch.max(logits.data, 1)

            if torch.cuda.is_available():
                correct += (predicted.detach().cpu().numpy() == target.detach().cpu().numpy()).sum()
            else:
                correct += (predicted == target).sum()

            pred_list.append(predicted.detach().cpu().numpy())
            label_list.append(target.detach().cpu().numpy())

    preds = np.concatenate(pred_list, axis=0)
    labels = np.concatenate(label_list, axis=0)

    f1_total = f1_score(labels, preds, labels=[0, 1], average=None)

    epoch_accuracy = 100 * float(correct) / preds.shape[0]
    if epoch_accuracy > accuracy:
        accuracy = epoch_accuracy
        torch.save(model.state_dict(), param_file_name)
        print(['Saved @ ' + str(accuracy) + '%'])
    print('Val accuracy: {:.4f}'.format(epoch_accuracy))
    print(f'Val F1: {f1_total[0]}, {f1_total[1]}')
    return accuracy


def test():
    model.load_state_dict(torch.load(param_file_name, map_location='cpu'))
    model.eval()
    correct = 0
    pred_list, label_list = [], []
    with torch.no_grad():
        for batch_idx, (_, target, feature, _, _, _) in enumerate(test_loader):
            target, feature = target.to(device), feature.to(device)
            logits = model(feature)[0]

            _, predicted = torch.max(logits.data, 1)

            if torch.cuda.is_available():
                correct += (predicted.detach().cpu().numpy() == target.detach().cpu().numpy()).sum()
            else:
                correct += (predicted == target).sum()
            pred_list.append(predicted.detach().cpu().numpy())
            label_list.append(target.detach().cpu().numpy())

    preds = np.concatenate(pred_list, axis=0)
    labels = np.concatenate(label_list, axis=0)

    f1_total = f1_score(labels, preds, labels=[0, 1], average=None)
    epoch_accuracy = 100 * float(correct) / preds.shape[0]
    print('Test accuracy: {:.4f}'.format(epoch_accuracy))
    print(f'Test F1: {f1_total[0]}, {f1_total[1]}')
    perf_dict = {'accuracy': epoch_accuracy, 'f1': f1_total}
    with open(perf_dict_name, 'wb') as f:
        pkl.dump(perf_dict, f, protocol=pkl.HIGHEST_PROTOCOL)
    return
    

if __name__ == "__main__":
    if learn_rep:
        accuracy = 0
        for epoch in range(1, num_epochs + 1):
            train(epoch)
            accuracy = val(accuracy)
        test()
    print(exp_name)
