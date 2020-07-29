import os
import torch
import torch.utils.data
import pickle as pkl
from torch import nn, optim
from tqdm import tqdm
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse

from networks import HSICClassifier, MLP1Layer
from EEG.datasets import init_datasets
from EEG.train_utils import get_device, run_params
from EEG.feature_utils import feature_names_len_from_subset

# experiment parameters
file_name = "1slice_DS_frequency_lambda20*0.25_lsb_remNrem_20_73"
task = 'rem_nrem'
cuda_id = 0
features_subset = ['frequency']
balanced_dataset = True

# training parameters
batch_size = 32
num_epochs = 40
lr = 0.0003
rep_size = 512

feature_opt = run_params(features_subset)
torch.manual_seed(44)
device = get_device(cuda_id)
file_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_models', file_name)

train_loader, val_loader, test_loader = init_datasets(task=task, balanced_dataset=balanced_dataset, batch_size=batch_size,
                                                      normalize_signals=True, features_subset=features_subset)

main_model = HSICClassifier(num_classes=2, feature_opt=feature_opt, gap_norm_opt='batch_norm',
                            feature_len=train_loader.dataset.feature_len, in_channels=2).to(device)

main_model.load_state_dict(torch.load(os.path.join(file_dir, f'{file_name}_params.pkl'), map_location='cpu'))
main_model.eval()

features_len, _ = feature_names_len_from_subset(features_subset)
feature_predictor = MLP1Layer(in_size=rep_size, hidden_size=rep_size, out_size=features_len).to(device)
optimizer = optim.Adam(feature_predictor.parameters(), lr=lr, weight_decay=1e-6)
criterion = nn.MSELoss()


def train_predict_features(epoch):
    feature_predictor.train()
    train_loss = 0
    for batch_idx, (signals, _, _, features) in enumerate(tqdm(train_loader)):

        signals, features = signals.to(device), features.to(device)
        optimizer.zero_grad()
        _, _, gap = main_model(x=signals, features=features)
        features_pred = feature_predictor(gap)
        loss = criterion(features, features_pred)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset)}')
    return


def val_predict_features():
    feature_predictor.eval()
    val_loss = 0
    r2 = 0
    r2_list = []
    with torch.no_grad():
        for batch_idx, (signals, _, _, features) in enumerate(val_loader):
            signals, features = signals.to(device), features.to(device)
            _, _, gap = main_model(x=signals, features=features)
            features_pred = feature_predictor(gap)
            pred_errors = mse(features.detach().cpu(), features_pred.detach().cpu(), multioutput='raw_values')
            val_loss += pred_errors
            curr_r2 = r2_score(features.detach().cpu(), features_pred.detach().cpu())
            r2 += curr_r2
            r2_list.append(curr_r2)

    val_loss /= val_loader.__len__()
    r2 /= val_loader.__len__()

    r2_list.append(r2)

    print('====> Val set MSE loss: {:.5f}'.format(val_loss.mean()))
    print('====> Val set R^2: {:.5f}'.format(r2))

    return r2_list


def test_predict_features():
    feature_predictor.eval()
    r2 = 0
    with torch.no_grad():
        for batch_idx, (signals, _, _, features) in enumerate(test_loader):
            signals, features = signals.to(device), features.to(device)
            _, _, gap = main_model(x=signals, features=features)
            features_pred = feature_predictor(gap)
            r2 += r2_score(features.detach().cpu(), features_pred.detach().cpu())

    r2 /= test_loader.__len__()
    print(f'====> Test set R^2: {r2}')
    with open(os.path.join(file_dir, f'{file_name}_test_r2.pkl'), 'wb') as handle:
        pkl.dump(r2, handle, protocol=pkl.HIGHEST_PROTOCOL)

    return


if __name__ == "__main__":

    r2_list = []
    for epoch in range(num_epochs):
        train_predict_features(epoch)
        r2_list = val_predict_features()

    test_predict_features()
