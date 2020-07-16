import os
import numpy as np
import torch
import torch.utils.data
import pandas as pd
import pickle as pkl
from torch import nn, optim
from tqdm import tqdm
from sklearn.metrics import r2_score, f1_score, precision_recall_fscore_support
from sklearn.metrics import mean_squared_error as mse

from ECG.train.datasets import GapDataset, create_dataloaders
from networks import HSICClassifier, MLP2Layer, MLP1Layer
from ECG.train.train_utils import get_device, generate_gap
import ECG.feature_utils as futil

cuda_id = 1

file_name = 'lambda_500_rr_balanced_NAF'
feature_subset = 'rr'
torch.manual_seed(42)
included_feature_set = 'rr'
excluded_feature_set = 'p_wave'
need_rep2label = True

naf = True
batch_size = 32
lr_rep2label = 0.0005
num_epochs_rep2label = 40
lr = 1e-3
num_epochs = 40
num_classes = 3
rep_size = 512
oversample_weights = '50'

print(f'{file_name}: Starting post train analysis')
file_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),  'saved_models', file_name)

is_baseline = 'baseline' in file_name.lower()
feature_opt = 'None' if is_baseline else 'HSIC+Concat'

device = get_device(cuda_id)
base_dir = ".."

if not os.path.exists(file_dir):
    os.mkdir(file_dir)

feature_subset_dataloader = feature_subset  # if feature_subset == 'all_fcnet' else 'all'

train_loader, val_loader, test_loader = create_dataloaders(batch_size=batch_size, feature_subset=feature_subset_dataloader,
                                                           feature_opt='HSIC+Concat', naf=naf)

haifa_model = HSICClassifier(num_classes=num_classes, feature_len=train_loader.dataset.feature_len,
                             feature_opt=feature_opt, gap_norm_opt='batch_norm').to(device)

haifa_model.load_state_dict(torch.load(os.path.join(file_dir, f'{file_name}_params.pkl'), map_location='cpu'))

haifa_model.eval()
correct = 0
pred_list = []
label_list = []

# test procedure
with torch.no_grad():
    for batch_idx, (data, target, _, _, sig_names, features_rep) in enumerate(test_loader):
        data, target, features_rep = data.to(device), target.to(device), features_rep.to(device)

        logits, _, gap = haifa_model(data, features_rep)
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

results_dict = {}
results_dict['accuracy'] = 100 * float(correct) / test_loader.dataset.__len__()
results_dict['f1_normal'] = f1_total[0]
results_dict['f1_af'] = f1_total[1]
results_dict['f1_other'] = f1_total[2]

for i,l in enumerate(['normal', 'af','other']):
    results_dict[f'precision_{l}'] = precision_recall_fscore_support(labels, preds, labels=[0, 1, 2], average=None)[0][i]
    results_dict[f'recall_{l}'] = precision_recall_fscore_support(labels, preds, labels=[0, 1, 2], average=None)[1][i]

print('Test accuracy: {:.4f}'.format(results_dict['accuracy']))
print(f'Test F1: {f1_total[0]}, {f1_total[1]}, {(f1_total[2])}')

with open(os.path.join(file_dir, f'{file_name}_test_res.pkl'), 'wb') as f:
    pkl.dump(results_dict, f, protocol=pkl.HIGHEST_PROTOCOL)

# Reinstantiate the test loader to remove its sampler
test_loader = torch.utils.data.DataLoader(dataset=test_loader.dataset, batch_size=batch_size, shuffle=False, sampler=None)

gap_train, gap_val, gap_test = generate_gap(train_loader, val_loader, test_loader, model=haifa_model,
                                            file_dir=file_dir, file_name=file_name, device=device)

print('Gap generation complete')

num_of_iteration = len(train_loader.dataset) // batch_size
num_of_test_iteration = len(val_loader.dataset) / batch_size
criterion = nn.MSELoss()


def train_predict_features(epoch, subset):
    feature_predictor.train()
    train_loss = 0
    for batch_idx, (_, _, _, real_features, signal_names, _) in enumerate(tqdm(train_loader)):
        real_features = real_features[:, subset].to(device)
        optimizer.zero_grad()

        # todo: predicting real features[idx], implement for val and test

        gap = torch.stack([torch.from_numpy(gap_train[name]).to(device) for name in signal_names])
        features_pred = feature_predictor(gap)
        loss = criterion(real_features, features_pred)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def val_predict_features(subset, best_r2):
    feature_predictor.eval()
    test_loss = 0
    r2_list = []
    with torch.no_grad():
        for batch_idx, (_, _, _, real_features, signal_names, _) in enumerate(val_loader):
            real_features = real_features[:, subset].to(device)

            gap = torch.stack([torch.from_numpy(gap_val[name]).to(device) for name in signal_names])
            gap = gap[:, :rep_size]
            features_pred = feature_predictor(gap)
            pred_errors = mse(real_features.detach().cpu(), features_pred.detach().cpu(), multioutput='raw_values')
            test_loss += pred_errors
            curr_r2 = r2_score(real_features.detach().cpu(), features_pred.detach().cpu())
            r2_list.append(curr_r2)

    test_loss /= num_of_test_iteration
    r2 = np.mean(r2_list)

    if r2 > best_r2:
        best_r2 = r2
        print(f'Saved @ {r2}')
        torch.save(feature_predictor.state_dict(), os.path.join(file_dir, f'{file_name}_rep2features_params.pkl'))

    print('====> Val set MSE loss: {:.5f}'.format(test_loss.mean()))
    print('====> Val set R^2: {:.5f}'.format(r2))

    return best_r2


def test_predict_features(subset, result_type):
    feature_predictor.eval()
    feature_predictor.load_state_dict(
        torch.load(os.path.join(file_dir, f'{file_name}_rep2features_params.pkl'), map_location='cpu'))
    r2_list = []
    with torch.no_grad():
        for batch_idx, (_, _, _, real_features, signal_names, _) in enumerate(test_loader):
            real_features = real_features[:, subset].to(device)

            gap = torch.stack([torch.from_numpy(gap_test[name]).to(device) for name in signal_names])
            gap = gap[:, :rep_size]
            features_pred = feature_predictor(gap)
            curr_r2 = r2_score(real_features.detach().cpu(), features_pred.detach().cpu())
            r2_list.append(curr_r2)

    r2 = np.mean(r2_list)

    print('====> Test set R^2: {:.5f}'.format(r2))

    with open(os.path.join(file_dir, f'{file_name}_test_r2{result_type}_fixed.pkl'), 'wb') as handle:
        pkl.dump(r2, handle, protocol=pkl.HIGHEST_PROTOCOL)

    return

r2_list = []
subset_list = []


def update_subset_list(subset_type, subset_list, included=True):

    if (not included) and (subset_type == 'None'):
        return

    status = 'included' if included else 'excluded'

    if 'rr' in subset_type:
        subset = futil.rr_feature_names
        result_type = f'_rr_{status}'
    elif 'p_wave' in subset_type:
        subset = futil.p_wave_feature_names
        result_type = f'_p_wave_{status}'
    else:
        if included:
            subset = list(train_loader.dataset.real_features)
            result_type = ''
    subset_list.append(subset)
    return result_type

result_type_list = []
result_type_list.append(update_subset_list(subset_type=included_feature_set, subset_list=subset_list, included=True))
result_type_list.append(update_subset_list(subset_type=excluded_feature_set, subset_list=subset_list, included=False))


def col_names_to_idx(subset, df):
    return[df.columns.get_loc(col) for col in subset]


curr_df = train_loader.dataset.real_features

for index, subset in enumerate(subset_list):

    num_features2predict = len(subset)
    feature_predictor = MLP2Layer(in_size=rep_size, hidden_size1=128, hidden_size2=128, out_size=num_features2predict).to(device)
    optimizer = optim.Adam(feature_predictor.parameters(), lr=lr, weight_decay=1e-6)
    best_r2 = -1e8
    for epoch in range(1, num_epochs + 1):
        train_predict_features(epoch, col_names_to_idx(subset, curr_df))
        best_r2 = val_predict_features(best_r2=best_r2, subset=col_names_to_idx(subset, curr_df))

    test_predict_features(col_names_to_idx(subset, curr_df), result_type_list[index])

# From here and following is taken from rep_to_label.py
if need_rep2label:
    # load test gap from run dir
    with open(os.path.join(file_dir, f'{file_name}_ugap_test.pkl'), 'rb') as handle:
        test_gaps = pkl.load(handle)

    with open(os.path.join(file_dir, f'{file_name}_ugap_train.pkl'), 'rb') as handle:
        train_gaps = pkl.load(handle)

    with open(os.path.join(file_dir, f'{file_name}_ugap_val.pkl'), 'rb') as handle:
        val_gaps = pkl.load(handle)

    def train_rep2label(epoch, optimizer, model, train_loader, criterion, gaps=train_gaps):
        model.train()
        train_loss = 0
        for batch_idx, (signal_names, labels) in enumerate(train_loader):
        # for batch_idx, (_, labels, _, _, signal_names, _) in enumerate(train_loader):
            labels = labels.to(device)
            optimizer.zero_grad()
            gap = torch.stack([torch.from_numpy(gaps[name]).to(device) for name in signal_names])
            logits = model(gap)
            loss = criterion(logits, labels)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))

        return


    def val_rep2label(best_acc, model, criterion, gaps=test_gaps):
        model.eval()
        correct, val_loss = 0, 0
        pred_list, label_list = [], []
        with torch.no_grad():
            for batch_idx, (signal_names, labels) in enumerate(train_loader):

                labels = labels.to(device)

                gap = torch.stack([torch.from_numpy(gaps[name]).to(device) for name in signal_names])

                gap = gap[:, :rep_size]
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
                torch.save(model.state_dict(), os.path.join(file_dir, f'{file_name}_rep2label_params.pkl'))
                print(['Saved @  ' + str(epoch_accuracy) + '%'])

        val_loss /= num_of_val_iteration

        print('====> Val set Accuracy: {:.5f}'.format(epoch_accuracy))
        print(f'====> Val set F1: {f1_total[0]}, {f1_total[1]}, {f1_total[2]}')
        return best_acc


    def test_rep2label(model, criterion, gaps=test_gaps):
        model.load_state_dict(torch.load(os.path.join(file_dir, f'{file_name}_rep2label_params.pkl'), map_location='cpu'))
        model.eval()
        correct, test_loss = 0, 0
        pred_list, label_list = [], []
        with torch.no_grad():
            for batch_idx, (signal_names, labels) in enumerate(train_loader):

                labels = labels.to(device)
                gap = torch.stack([torch.from_numpy(gaps[name]).to(device) for name in signal_names])

                gap = gap[:, :rep_size]
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

        test_loss /= num_of_test_iteration

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
    for idx in range(5):
        train_dataset = GapDataset(mode="train", idx=idx, naf=naf, oversample=oversample_weights)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False,
                                                   sampler=train_dataset.sampler)

        val_dataset = GapDataset(mode="val", idx=idx, naf=naf, oversample=oversample_weights)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,
                                                 sampler=val_dataset.sampler)

        test_dataset = GapDataset(mode="test", idx=idx, naf=naf, oversample=oversample_weights)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                                  sampler=test_dataset.sampler)

        haifa_model = MLP1Layer(in_size=rep_size, hidden_size=rep_size, out_size=num_classes).to(device)

        num_of_iteration = (len(train_dataset) // batch_size) + 1
        num_of_val_iteration = (len(val_dataset) // batch_size) + 1
        num_of_test_iteration = (len(test_dataset) // batch_size) + 1
        optimizer = optim.Adam(haifa_model.parameters(), lr=lr_rep2label, weight_decay=1e-6)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0
        for epoch in range(1, num_epochs_rep2label + 1):
            train_rep2label(epoch, optimizer, haifa_model, train_loader, criterion, gaps=test_gaps)
            best_acc = val_rep2label(best_acc, haifa_model, criterion, gaps=test_gaps)

        rep2label_perf[idx] = test_rep2label(haifa_model, criterion, gaps=test_gaps)
        val_acc_rep2label.append(best_acc)

    with open(os.path.join(file_dir, f'{file_name}_rep2label_perf_ln.pkl'), 'wb') as handle:
        pkl.dump(rep2label_perf, handle, protocol=pkl.HIGHEST_PROTOCOL)

    with open(os.path.join(file_dir, f'{file_name}_val_acc_rep2label_ln.pkl'), 'wb') as handle:
        pkl.dump(val_acc_rep2label, handle, protocol=pkl.HIGHEST_PROTOCOL)


    def create_rep2label_datasets(naf, gaps):

        datasets = {}
        for idx in range(5):
            datasets[idx] = {}

            for mode in ['train', 'val', 'test']:

                dataset_path = os.path.join(os.getcwd(), 'data', 'held_out_data', f'{mode}_{idx}.pkl')
                labels = pd.read_pickle(dataset_path)

                if naf:
                    labels.loc[labels.target == 2, 'target'] = 0
                labels = labels[labels.target != 3]

                labels = labels.reset_index()
                labels.drop(columns={'index'}, inplace=True)
                dataset_size = len(labels)

                signals = list(labels['signal'])
                targets = np.array(list(labels['target']))

                data = np.vstack([gaps[sig] for sig in signals])

                datasets[idx][mode] = (data, targets)

        return datasets

print(f'{file_name}: Finished post train analysis')
