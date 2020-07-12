from torch import device, cuda
import os
import pickle as pkl
import glob


def get_device(cuda_id):
    if not isinstance(cuda_id, str):
        cuda_id = str(cuda_id)
    device_ = device("cuda:" + cuda_id if cuda.is_available() else "cpu")
    return device_


def unify_gap(mode, file_dir, file_name):
    gap_unified = {}
    for file in os.listdir(file_dir):
        if file.startswith(f'{file_name}_gap_{mode}_'):
            file2read = os.path.join(file_dir, file)
            with open(file2read, 'rb') as h:
                gap_dict = pkl.load(h)
            gap_unified.update(gap_dict)
            os.remove(file2read)

    for key, value in gap_unified.items():
        gap_unified[key] = value.cpu().detach().numpy()

    with open(os.path.join(file_dir, f'{file_name}_ugap_{mode}.pkl'), 'wb') as h:
        pkl.dump(gap_unified, h, protocol=pkl.HIGHEST_PROTOCOL)

    return gap_unified


def extract_gap(mode, loader, model, file_dir, file_name, device):
    gap_dict = {}
    cam_dict = {}

    for batch_idx, (data, _, _, _, sig_names, features_rep) in enumerate(loader):
        data, features_rep = data.to(device), features_rep.to(device)
        _, cam, gap = model(data, features_rep)

        gap = gap[:, :model.activation_size]
        update_gap_dict(sig_names, gap, gap_dict, cam, cam_dict)

        with open(os.path.join(file_dir, f'{file_name}_gap_{mode}_{batch_idx}.pkl'), 'wb') as handle:
            pkl.dump(gap_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

        gap_dict = {}
        cam_dict = {}


def update_gap_dict(sig_names, gap, gap_dict, cam, cam_dict):
    for i in range(len(sig_names)):
        gap_dict[sig_names[i]] = gap[i]
        cam_dict[sig_names[i]] = cam[i, 2]


def generate_gap(train_loader, val_loader, test_loader, model, file_dir, file_name, device):
    extract_gap('train', train_loader, model=model, file_dir=file_dir, file_name=file_name, device=device)
    extract_gap('val', val_loader, model=model, file_dir=file_dir, file_name=file_name, device=device)
    extract_gap('test', test_loader, model=model, file_dir=file_dir, file_name=file_name, device=device)
    print('extraction done')

    gap_train = unify_gap('train', file_dir=file_dir, file_name=file_name)
    gap_val = unify_gap('val', file_dir=file_dir, file_name=file_name)
    gap_test = unify_gap('test', file_dir=file_dir, file_name=file_name)

    for f in glob.glob(f'{file_dir}/{file_name}_gap*'):
        os.remove(f)

    return gap_train, gap_val, gap_test
