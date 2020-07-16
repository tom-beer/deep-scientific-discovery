from __future__ import absolute_import, division, print_function

from torch import device, cuda
import os
import pickle as pkl
import glob
import numpy as np


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


"""
learning_rate_schedulers.py
---------------------------
This module provide classes and functions for managing learning rate schedules.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""


class AnnealingRestartScheduler(object):

    """
    Cyclical learning rate decay with warm restarts and cosine annealing.
    Reference: https://arxiv.org/pdf/1608.03983.pdf
    """

    def __init__(self, lr_min, lr_max, steps_per_epoch, lr_max_decay, epochs_per_cycle, cycle_length_factor):

        # Set parameters
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.steps_per_epoch = steps_per_epoch
        self.lr_max_decay = lr_max_decay
        self.epochs_per_cycle = epochs_per_cycle
        self.cycle_length_factor = cycle_length_factor

        # Set attributes
        self.lr = self.lr_max
        self.steps_since_restart = 0
        self.next_restart = self.epochs_per_cycle

    def on_batch_end_update(self):
        """Update at the end of each mini-batch."""
        # Update steps since restart
        self.steps_since_restart += 1

        # Update learning rate
        self.lr = self._compute_cosine_learning_rate()

    def on_epoch_end_update(self, epoch):
        """Check for end of current cycle, apply restarts when necessary."""
        if epoch + 1 == self.next_restart:
            self.steps_since_restart = 0
            self.epochs_per_cycle = np.ceil(self.epochs_per_cycle * self.cycle_length_factor)
            self.next_restart += self.epochs_per_cycle
            self.lr_max *= self.lr_max_decay

    def _compute_cosine_learning_rate(self):
        """Compute cosine learning rate decay."""
        # Compute the cycle completion factor
        fraction_complete = self._compute_fraction_complete()

        # Compute learning rate
        return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(fraction_complete * np.pi))

    def _compute_fraction_complete(self):
        """Compute the fraction of the cycle that is completed."""
        return self.steps_since_restart / (self.steps_per_epoch * self.epochs_per_cycle)


def exponential_step_decay(decay_epochs, decay_rate, initial_learning_rate, epoch):
    """Compute exponential learning rate step decay."""
    return initial_learning_rate * np.power(decay_rate, np.floor((epoch / decay_epochs)))

