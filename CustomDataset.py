import random

import numpy as np
import torch
from torch.utils.data import Dataset


class CustomTrainingDataset(Dataset):

    def __init__(self, D):
        self.D = D

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):
        context, action, reward = self.D[idx]
        return torch.Tensor(context), action, reward


class CustomHessianDataset(Dataset):

    def __init__(self, D, n_subsample=None):

        self.D = [(tuple_id, context, action, reward) for tuple_id, (context, action, reward) in enumerate(D)]
        if n_subsample is not None and n_subsample < len(self.D):
            self.D = random.sample(self.D, n_subsample)



    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):
        tuple_id, context, action, reward = self.D[idx]
        x = torch.concat([torch.Tensor([tuple_id]), torch.Tensor(context)])
        return x.to(torch.float), np.float32(reward)
