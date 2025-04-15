from monai.transforms import Transform
from torch.distributions import Beta
from torch import nn
import numpy as np
import torch

class MixupTransformd(Transform):
    """
    A dictionary-based wrapper for Mixup augmentation that applies to batches.
    This needs to be applied after batching, typically directly in the training loop.
    """
    def __init__(self, keys=("image", "label"), mix_beta=0.2, mixadd=False, prob=0.5):
        self.keys = keys
        self.mixup = Mixup(mix_beta=mix_beta, mixadd=mixadd)
        self.prob = prob
        
    def __call__(self, data):
        d = dict(data)
        if np.random.random() < self.prob:  # Apply with probability
            d[self.keys[0]], d[self.keys[1]] = self.mixup(d[self.keys[0]], d[self.keys[1]])
        return d

class Mixup(nn.Module):
    def __init__(self, mix_beta, mixadd=False):

        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)
        self.mixadd = mixadd

    def forward(self, X, Y, Z=None):

        bs = X.shape[0]
        n_dims = len(X.shape)
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device)
        X_coeffs = coeffs.view((-1,) + (1,)*(X.ndim-1))
        Y_coeffs = coeffs.view((-1,) + (1,)*(Y.ndim-1))
        
        X = X_coeffs * X + (1-X_coeffs) * X[perm]

        if self.mixadd:
            Y = (Y + Y[perm]).clip(0, 1)
        else:
            Y = Y_coeffs * Y + (1 - Y_coeffs) * Y[perm]
                
        if Z:
            return X, Y, Z

        return X, Y
