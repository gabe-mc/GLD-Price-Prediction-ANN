"""A single tensor class, creating and storing torch.tensor objects"""

import torch
import torch.utils.data as D


class Tensor(D.Dataset):
    """
    Class representing torch.tensor data. Is able to take DataFrame objects and convert them into Pytorch tensors.

    Instance Variables:
        features: takes a Panda DataFrame object of numerical features
        target: takes a single Panda DataFrame column of numerical targets
        len: length of tensor
    """

    def __init__(self, features, target):
        self.features = torch.tensor(features.values)
        self.target = torch.tensor(target.values)
        self.len = self.features.shape[0]

    def __getitem__(self, index):
        return self.features[index], self.target[index]

    def __len__(self):
        return self.len
