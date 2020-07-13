import os
import numpy as np
import torch
import torch.utils.data
import pickle as pkl
from torch import nn, optim
from tqdm import tqdm
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse

from networks import HaifaNetVPT, Rep2Label, OneFC
from EEG.datasets import init_datasets, GapDataset, change_sampler
from EEG.train_utils import generate_gap_internal, col_names_to_idx
from EEG.features import feature_names_len_from_subset
from EEG.train_utils import get_device, calc_save_perf_metrics, run_params

file_name = "1slice_DS_Baseline_balanced"

task = 'rem_nrem'  # ['rem_nrem', 'wake_rem', 'all']
cuda_id = 1
normalize_signals = True
gap_norm_opt = 'batch_norm'
feature_opt = 'None'  # 'HSIC+Concat'
features_subset = ['frequency']  # ['frequency', 'spindle']
oversample = True

# Things we don't really change:
batch_size = 32
num_epochs, num_epochs_rep2label = 15, 40
lr_rep2label = 0.0005
lr_features = 0.0003
num_patients = 1e8
num_ch = 2
low_sp = False
need_predict_features = True
need_rep2label = False
_, signal_len, one_slice, dataset_dir = run_params(file_name, features_subset, def_feature_opt='HSIC+Concat', task=task)
num_classes = 2 if 'rem' in task.lower() else 5
torch.manual_seed(44)

device = get_device(cuda_id)
print(f'device: {device}')

file_dir = os.path.join('saved_models', file_name)


# def check_hsic(loader, model):
#     model.eval()
#     feature_hsic_list = []
#     sigma_gap = 1
#     for batch_idx, (signals, _, _, features) in enumerate(tqdm(loader)):
#         signals, features = signals.to(device), features.to(device)
#
#         _, _, gap = model(signals, features)
#
#         med_dist_gap = np.median(pdist(gap.detach().cpu().numpy(), metric='euclidean').reshape(-1, 1))
#         sigma_gap = np.maximum(0.7 * sigma_gap + 0.3 * med_dist_gap, 0.005)
#         gap = gap[:, :model.activation_size]
#         hsic_features = HSIC(gap, features, kernelX='Gaussian', kernelY='Gaussian',
#                              sigmaX=sigma_gap, sigmaY=train_loader.dataset.med_dist, device=device)
#         feature_hsic_list.append(hsic_features)
#     return feature_hsic_list


if __name__ == "__main__":

    print(f'{file_name} started PTA with multi')
    train_loader, val_loader, test_loader = init_datasets(num_patients=num_patients, dataset_dir=dataset_dir, task=task,
                                                          one_slice=one_slice, batch_size=batch_size, conv_d=1,
                                                          oversample=False, normalize_signals=normalize_signals,
                                                          features_subset=features_subset, num_ch=num_ch, low_sp=low_sp)

    haifa_model = HaifaNetVPT(num_classes=num_classes, signal_len=signal_len, feature_opt=feature_opt, gap_norm_opt=gap_norm_opt,
                              feature_len=train_loader.dataset.feature_len, in_channels=num_ch).to(device)

    haifa_model.load_state_dict(torch.load(os.path.join(file_dir, f'{file_name}_params.pkl'), map_location='cpu'))

    # hsic_features = check_hsic(train_loader, haifa_model)
    # print(f'hsic_features: {np.mean(hsic_features)}')

    # generate gap if ugap does not exist
    gap_train, gap_val, gap_test, rep_size = generate_gap_internal(train_loader, val_loader, test_loader, haifa_model, file_dir, file_name, device)


    train_loader = change_sampler(train_loader, new_state=oversample)
    val_loader = change_sampler(val_loader, new_state=oversample)
    test_loader = change_sampler(test_loader, new_state=oversample)

    if need_predict_features:
        # r2_list = train_test_predict_features(features_subset, train_loader, val_loader, test_loader, device,
        #                                       gap_train, gap_val, gap_test, rep_size, file_name)

        def train_predict_features(epoch, subset):
            feature_predictor.train()
            train_loss = 0
            for batch_idx, (_, _, signal_names, features) in enumerate(tqdm(train_loader)):
                features = features[:, subset].to(device)
                optimizer.zero_grad()
                gap = torch.stack([torch.from_numpy(gap_train[name]).to(device) for name in signal_names])
                features_pred = feature_predictor(gap)
                loss = criterion(features, features_pred)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            print('====> Epoch: {} Average loss: {:.4f}'.format(
                epoch, train_loss / len(train_loader.dataset)))

            return

        def val_predict_features(r2_list, subset):
            num_iterations = val_loader.dataset.__len__() // val_loader.batch_size
            feature_predictor.eval()
            val_loss = 0
            r2 = 0
            r2_list = []
            with torch.no_grad():
                for batch_idx, (_, _, signal_names, features) in enumerate(val_loader):
                    features = features[:, subset].to(device)

                    gap = torch.stack([torch.from_numpy(gap_val[name]).to(device) for name in signal_names])
                    gap = gap[:, :rep_size]
                    features_pred = feature_predictor(gap)
                    pred_errors = mse(features.detach().cpu(), features_pred.detach().cpu(), multioutput='raw_values')
                    val_loss += pred_errors
                    curr_r2 = r2_score(features.detach().cpu(), features_pred.detach().cpu())
                    r2 += curr_r2
                    r2_list.append(curr_r2)

            val_loss /= num_iterations
            r2 /= num_iterations

            r2_list.append(r2)

            print('====> Val set MSE loss: {:.5f}'.format(val_loss.mean()))
            print('====> Val set R^2: {:.5f}'.format(r2))
            print('====> Val set R^2 from list: {:.5f}'.format(np.mean(r2_list)))


            return r2_list

        def test_predict_features(subset):
            num_iterations = test_loader.dataset.__len__() // test_loader.batch_size
            feature_predictor.eval()
            r2 = 0
            with torch.no_grad():
                for batch_idx, (_, _, signal_names, features) in enumerate(test_loader):
                    features = features[:, subset].to(device)
                    gap = torch.stack([torch.from_numpy(gap_test[name]).to(device) for name in signal_names])
                    gap = gap[:, :rep_size]
                    features_pred = feature_predictor(gap)
                    r2 += r2_score(features.detach().cpu(), features_pred.detach().cpu())

            r2 /= num_iterations
            print('====> Test set R^2: {:.5f}'.format(r2))
            with open(os.path.join(file_dir, f'{file_name}_test_r2.pkl'), 'wb') as handle:
                pkl.dump(r2, handle, protocol=pkl.HIGHEST_PROTOCOL)

            return

    r2_list = []

    features_len, subset = feature_names_len_from_subset(features_subset)
    # feature_predictor = multi_FC_FeatureNet(rep_size=rep_size, out_size=features_len).to(device)
    feature_predictor = OneFC(in_size=rep_size, num_classes=features_len).to(device)
    optimizer = optim.Adam(feature_predictor.parameters(), lr=lr_features, weight_decay=1e-6)
    criterion = nn.MSELoss()
    curr_df = train_loader.dataset.features[subset]

    for epoch in range(1, num_epochs + 1):
        train_predict_features(epoch, col_names_to_idx(subset, curr_df))
        r2_list = val_predict_features(r2_list, col_names_to_idx(subset, curr_df))

    test_predict_features(col_names_to_idx(subset, curr_df))

if need_rep2label:
    def train_rep2label(epoch, optimizer, model, train_loader, criterion, gaps):
        model.train()
        train_loss = 0
        for batch_idx, (signal_names, labels) in tqdm(enumerate(train_loader)):
            labels = labels.to(device)
            optimizer.zero_grad()
            gap = torch.stack([torch.from_numpy(gaps[name]).to(device) for name in signal_names])
            logits = model(gap)
            loss = criterion(logits, labels)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        scheduler.step()

        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        return


    def val_test_rep2label(model, loader, gaps, mode, file_name, best_acc=None):
        assert mode in ['val', 'test']
        if mode == 'test':
            model.load_state_dict(
                torch.load(os.path.join('saved_models', file_name, f'{file_name}_rep2label_params.pkl'), map_location='cpu'))
        model.eval()
        correct = 0
        pred_list, label_list = [], []
        with torch.no_grad():
            for batch_idx, (signal_names, labels) in enumerate(loader):

                labels = labels.to(device)

                gap = torch.stack([torch.from_numpy(gaps[name]).to(device) for name in signal_names])

                gap = gap[:, :rep_size]
                logits = model(gap)
                _, predicted = torch.max(logits.data, 1)

                if torch.cuda.is_available():
                    correct += (predicted.detach().cpu().numpy() == labels.detach().cpu().numpy()).sum()
                else:
                    correct += (predicted == labels).sum()

                pred_list.append(predicted.detach().cpu().numpy())
                label_list.append(labels.detach().cpu().numpy())

            preds = np.concatenate(pred_list, axis=0)
            labels = np.concatenate(label_list, axis=0)

            epoch_accuracy = 100 * float(correct) / loader.dataset.__len__()

            if mode == 'val' and epoch_accuracy >= best_acc:
                best_acc = epoch_accuracy
                torch.save(model.state_dict(), os.path.join('saved_models', file_name, f'{file_name}_rep2label_params.pkl'))
                print(['Saved @  ' + str(epoch_accuracy) + '%'])

        results_dict = calc_save_perf_metrics(labels, preds, epoch_accuracy, epoch, mode=mode, file_name=file_name, save=False)
        if mode == 'val':
            return best_acc
        else:
            return results_dict


    rep2label_perf = {}
    val_acc_rep2label = []
    assert dataset_dir == 'filtered0.05'  # 5-fold CV set extracted only for this folder...
    print(f'{file_name} starts rep2label')
    for idx in range(5):  # num CV splits
        print(f'{file_name} is loading GapDataset')

        train_dataset = GapDataset(mode="train", idx=idx)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

        val_dataset = GapDataset(mode="val", idx=idx)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

        test_dataset = GapDataset(mode="test", idx=idx)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

        rep2label_model = Rep2Label(num_classes=num_classes, in_size=rep_size).to(device)

        optimizer = optim.Adam(rep2label_model.parameters(), lr=lr_rep2label, weight_decay=1e-6)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, eta_min=lr_rep2label / 100, T_max=num_epochs_rep2label)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0
        for epoch in range(1, num_epochs_rep2label + 1):
            train_rep2label(epoch, optimizer, rep2label_model, train_loader, criterion=criterion, gaps=gap_train)
            best_acc = val_test_rep2label(model=rep2label_model, loader=val_loader, gaps=gap_train, mode='val',
                                          file_name=file_name, best_acc=best_acc)

        rep2label_perf[idx] = val_test_rep2label(model=rep2label_model, loader=test_loader, gaps=gap_train, mode='test',
                                                 file_name=file_name, best_acc=best_acc)
        val_acc_rep2label.append(best_acc)

        with open(os.path.join(file_dir, f'{file_name}_rep2label_perf.pkl'), 'wb') as handle:
            pkl.dump(rep2label_perf, handle, protocol=pkl.HIGHEST_PROTOCOL)

        with open(os.path.join(file_dir, f'{file_name}_val_acc_rep2label.pkl'), 'wb') as handle:
            pkl.dump(val_acc_rep2label, handle, protocol=pkl.HIGHEST_PROTOCOL)




print(f'{file_name}: Finished post train analysis')
