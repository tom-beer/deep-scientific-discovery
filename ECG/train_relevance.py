import os
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from tqdm import tqdm
from sklearn.metrics import f1_score
from networks import FC_FeatureNet
from ECG.train.datasets import ECGDataset
from ECG.feature_utils import update_rep_dict
import pickle as pkl


feature_subset = 'all'
run_name = 'test' # f'{feature_subset}_features_to_labels_NAF'

cuda_id = str(2)
learn_rep = True
extract_rep = False
naf = True
oversample = '50'

torch.manual_seed(42)

ds = True
signal_len = 5400
batch_size = 48


extracted_rep_file_name = f'{run_name}.pkl'
param_file_name = f'{run_name}_params.pkl'
perf_dict_name = f'{run_name}_perf.pkl'
print(run_name)

num_epochs = 40
in_channels_ = 1
num_classes = 2
lr = 0.005
is_cuda = True
if 'Documents' in os.getcwd():  # if running from desktop
    is_cuda = False
device = torch.device("cuda:"+cuda_id if is_cuda else "cpu")
feature_opt = 'Concat'
base_dir = ".."

train_dataset = ECGDataset("train", feature_subset=feature_subset, feature_opt=feature_opt,
                           oversample=oversample, naf=naf)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False,
                                           sampler=train_dataset.sampler)
val_dataset = ECGDataset("val", feature_subset=feature_subset, feature_opt=feature_opt,
                         oversample=oversample, naf=naf)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,
                                         sampler=val_dataset.sampler)
test_dataset = ECGDataset("test", feature_subset=feature_subset, feature_opt=feature_opt,
                          oversample=oversample, naf=naf)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                          sampler=test_dataset.sampler)

model = FC_FeatureNet(num_classes=num_classes, feature_len=train_dataset.feature_len).to(device)

num_of_iteration = len(train_dataset) // batch_size

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
criterion = nn.CrossEntropyLoss()


def extract_rep_dset(loader, rep_dict):
    for batch_idx, (_, target, feature, _, signal_names, _) in enumerate(tqdm(loader)):
        target, feature = target.to(device), feature.to(device)
        _, rep = model(feature)

        rep_dict = update_rep_dict(signal_names, rep, rep_dict)
    return rep_dict


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
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))
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
    if extract_rep:
        model.load_state_dict(torch.load(param_file_name, map_location='cpu'))
        rep_dict = {}
        rep_dict = extract_rep_dset(train_loader, rep_dict)
        rep_dict = extract_rep_dset(val_loader, rep_dict)
        rep_dict = extract_rep_dset(test_loader, rep_dict)
        with open(extracted_rep_file_name, 'wb') as handle:
            pkl.dump(rep_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)
        print(f"Done! Don't forget to move {extracted_rep_file_name} to data_dir")
    print(run_name)
