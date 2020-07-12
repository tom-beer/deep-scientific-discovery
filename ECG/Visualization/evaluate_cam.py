import torch
from ECG.Visualization.plot_utils import plot_cam
from networks import HaifaNetVPT
from ECG.train.custom_dataloader import CustomDataset

# lambda_500_rr_balanced_NAF
# 'BaselineDS_balanced_naf'
# 'lambda_500_all_naf_lmb'
# lambda_100_rr_naf_nonoise

# BaselineDS_naf_alpha1
# lambda_500_rr_naf_alpha1
model_name = 'BaselineDS_balanced_naf'

# model_name = "lambda_350_fixed_rr_50"
# model_name = 'lambda_500_rr_balanced_k15'
# model_name = 'lambda_900_rr_50_balanced_k15'
# model_name = 'lambda_10_rr_balanced_k15' # overfits
sin_alpha = 0
label = 'af'  # ['normal', 'af', 'other']
kernel_size = 3
plot_inds = (1500, 2000)

feature_subset = 'rr' if 'rr' in model_name else 'all'
feature_opt = 'None' if 'baseline' in model_name.lower() else 'HSIC+Concat'
print(feature_subset)
print(feature_opt)
gap_norm = 'batch_norm'
batch_size = 16
in_channels_ = 1
signal_len = 5400
device = "cpu"

if label == 'normal':
    # signal_list = ['A00012', 'A00035', 'A00042', 'A00059', 'A00072','A00095','A00104']
    signal_list = ['A00035']
    # signal_list = ['A01956', 'A01770']
    cam_target = 0
elif label == 'af':
    # signal_list = ['A03979', 'A08500', 'A00102', 'A00486', 'A00208', 'A00231', 'A00395', 'A00422', 'A00456', 'A00860','A00725']
    signal_list = ['A00486']
    # tosend - ['A00102', 'A03979', 'A00486', 'A00860']
    cam_target = 1


else:
    assert (label == 'other')
    # signal_list = ['A00285', 'A01418', 'A01474', 'A01525']
    signal_list = ['A08510']
    # signal_list = ['A00055', 'A00077', 'A00008', 'A00038']
    cam_target = 2


test_dataset = CustomDataset(mode="test", signal_len=signal_len, feature_subset=feature_subset, feature_opt=feature_opt,
                             permute_labels=False, downsample=True, oversample='none', is_baseline=False,
                             sin_alpha=sin_alpha)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
model = HaifaNetVPT(num_classes=3, signal_len=signal_len, feature_len=test_dataset.feature_len,
                    feature_opt=feature_opt, gap_norm_opt=gap_norm, kernel_size=kernel_size).to(device)
# model = torch.nn.DataParallel(feature_model, device_ids=[1, 2, 3])
param_filename = f'saved_models/{model_name}/{model_name}_params.pkl'
model.load_state_dict(torch.load(param_filename, map_location='cpu'))
model.eval()

for i in range(len(signal_list)):
    plot_cam(saved_model_name=model_name, signal_name=signal_list[i], plot_inds=plot_inds, test_loader=test_loader,
             model=model, cam_target=cam_target, label=label)
