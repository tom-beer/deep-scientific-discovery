import torch
import numpy as np
from sklearn.metrics import pairwise_distances as pdist


def pairwise_distances(x):
    x_distances = torch.sum(x**2, -1).reshape((-1, 1))
    return -2*torch.mm(x, x.t()) + x_distances + x_distances.t()


def kernelMatrixGaussian(x, sigma=1):
    pairwise_distances_ = pairwise_distances(x)
    gamma = -1.0 / (sigma ** 2)
    return torch.exp(gamma * pairwise_distances_)


def kernelMatrixLinear(x):
    return torch.matmul(x, x.t())


def HSIC(X, Y, kernelX="Gaussian", kernelY="Gaussian", sigmaX=1., sigmaY=1., device="cpu"):

    m, _ = X.shape

    K = kernelMatrixGaussian(X, sigmaX) if kernelX == "Gaussian" else kernelMatrixLinear(X)
    L = kernelMatrixGaussian(Y, sigmaY) if kernelY == "Gaussian" else kernelMatrixLinear(Y)

    H = (torch.eye(m) - 1.0/m * torch.ones((m, m))).to(device)

    Kc = torch.mm(H, torch.mm(K, H))

    HSIC = torch.trace(torch.mm(L, Kc)) / (m ** 2)  # it's supposed to be divided by m-1 but it's not robust
    # to cases of batch size of 1 so I changed to m. Shouldn't make a difference..
    return HSIC


class HSICLoss:
    def __init__(self, feature_opt, lambda_hsic, activation_size, device, decay_factor, external_feature_std):
        self.flag_calc_loss = 'hsic' in feature_opt.lower()
        self.lambda_hsic = lambda_hsic
        self.activation_size = activation_size
        self.sigma_gap = 1  # just an init, it will be overridden fast
        self.device = device
        self.decay_factor = decay_factor
        self.external_feature_std = external_feature_std

    def calc_loss(self, gap, feature):
        loss = torch.zeros(1)
        if self.flag_calc_loss:
            # calculate median distance between all pairs of points
            med_dist = np.median(pdist(gap.detach().cpu().numpy(), metric='euclidean').reshape(-1, 1))
            # calculate current kernel bandwidth as moving average of previous sigma and current median distance
            sigma_gap = np.maximum(self.decay_factor * self.sigma_gap +
                                   (1-self.decay_factor) * med_dist, 0.005)
            gap = gap[:, :self.activation_size]  # penalize only the latent representation, not the external features
            hsic_features = HSIC(gap, feature, kernelX='Gaussian', kernelY='Gaussian',
                                 sigmaX=sigma_gap, sigmaY=self.external_feature_std, device=self.device)
            loss = self.lambda_hsic * hsic_features
        return loss
