import os
import numpy as np
import torch
import torch.utils.data
import pickle as pkl
from torch import nn, optim
from tqdm import tqdm
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse

from networks import HSICClassifier, MLP2Layer
from ECG.train.datasets import create_dataloaders
from ECG.train.train_utils import get_device, update_subset_list

# experiment parameters
exp_name = 'main_task_lambda500_rr'
feature_subset = 'rr'
included_feature_set = 'rr'
excluded_feature_set = 'p_wave'
run_rep2label = False

# training parameters
torch.manual_seed(42)
cuda_id = 0
batch_size = 32
lr = 1e-3
num_epochs = 40
rep_size = 512

file_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),  'saved_models', exp_name)

is_baseline = 'baseline' in exp_name.lower()
feature_opt = 'None' if is_baseline else 'HSIC+Concat'

device = get_device(cuda_id)

if not os.path.exists(file_dir):
    os.mkdir(file_dir)

feature_subset_dataloader = feature_subset  # if feature_subset == 'all_fcnet' else 'all'

train_loader, val_loader, test_loader = create_dataloaders(batch_size=batch_size, feature_subset=feature_subset_dataloader,
                                                           feature_opt='HSIC+Concat', naf=True)

main_model = HSICClassifier(num_classes=2, feature_len=train_loader.dataset.feature_len,
                            feature_opt=feature_opt, gap_norm_opt='batch_norm', in_channels=1).to(device)

main_model.load_state_dict(torch.load(os.path.join(file_dir, f'{exp_name}_params.pkl'), map_location='cpu'))
main_model.eval()
criterion = nn.MSELoss()


def train_predict_features(epoch, subset):
    feature_predictor.train()
    train_loss = 0
    for batch_idx, (signal, _, features, real_features, signal_names, _) in enumerate(tqdm(train_loader)):
        signal, features, real_features = signal.to(device), features.to(device), real_features[:, subset].to(device)
        optimizer.zero_grad()
        _, _, gap = main_model(x=signal, features=features)
        features_pred = feature_predictor(gap)
        loss = criterion(real_features, features_pred)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')


def val_predict_features(subset, best_r2):
    feature_predictor.eval()
    test_loss = 0
    r2_list = []
    with torch.no_grad():
        for batch_idx, (signal, _, features, real_features, signal_names, _) in enumerate(val_loader):
            signal, features, real_features = signal.to(device), features.to(device), real_features[:, subset].to(device)

            _, _, gap = main_model(x=signal, features=features)
            features_pred = feature_predictor(gap)
            pred_errors = mse(real_features.detach().cpu(), features_pred.detach().cpu(), multioutput='raw_values')
            test_loss += pred_errors
            curr_r2 = r2_score(real_features.detach().cpu(), features_pred.detach().cpu())
            r2_list.append(curr_r2)

    test_loss /= len(test_loader)
    r2 = np.mean(r2_list)

    if r2 > best_r2:
        best_r2 = r2
        print(f'Saved @ {r2}')
        torch.save(feature_predictor.state_dict(), os.path.join(file_dir, f'{exp_name}_rep2features_params.pkl'))

    print(f'====> Val set MSE loss: {test_loss.mean():.5f}')
    print(f'====> Val set R^2: {r2:.5f}')

    return best_r2


def test_predict_features(subset, result_type):
    feature_predictor.eval()
    feature_predictor.load_state_dict(
        torch.load(os.path.join(file_dir, f'{exp_name}_rep2features_params.pkl'), map_location='cpu'))
    r2_list = []
    with torch.no_grad():
        for batch_idx, (signal, _, features, real_features, signal_names, _) in enumerate(test_loader):
            signal, features, real_features = signal.to(device), features.to(device), real_features[:, subset].to(device)

            _, _, gap = main_model(x=signal, features=features)
            features_pred = feature_predictor(gap)
            curr_r2 = r2_score(real_features.detach().cpu(), features_pred.detach().cpu())
            r2_list.append(curr_r2)

    r2 = np.mean(r2_list)

    print('====> Test set R^2: {:.5f}'.format(r2))

    with open(os.path.join(file_dir, f'{exp_name}_test_r2{result_type}_fixed.pkl'), 'wb') as handle:
        pkl.dump(r2, handle, protocol=pkl.HIGHEST_PROTOCOL)

    return

r2_list = []
subset_list = []

result_type_list = []
result_type_list.append(update_subset_list(subset_type=included_feature_set, subset_list=subset_list, included=True,
                                           real_features=list(train_loader.dataset.real_features)))
result_type_list.append(update_subset_list(subset_type=excluded_feature_set, subset_list=subset_list, included=False,
                                           real_features=list(train_loader.dataset.real_features)))


def col_names_to_idx(subset, df):
    return[df.columns.get_loc(col) for col in subset]


curr_df = train_loader.dataset.real_features

for index, subset in enumerate(subset_list):

    num_features2predict = len(subset)
    feature_predictor = MLP2Layer(in_size=rep_size, hidden_size1=128, hidden_size2=128, out_size=num_features2predict).to(device)
    optimizer = optim.Adam(feature_predictor.parameters(), lr=lr, weight_decay=1e-6)
    best_r2 = -1e8
    for epoch in range(num_epochs):
        train_predict_features(epoch, col_names_to_idx(subset, curr_df))
        best_r2 = val_predict_features(best_r2=best_r2, subset=col_names_to_idx(subset, curr_df))

    test_predict_features(col_names_to_idx(subset, curr_df), result_type_list[index])
