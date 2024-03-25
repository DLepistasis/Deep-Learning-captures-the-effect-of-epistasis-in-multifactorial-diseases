import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from imblearn.over_sampling import RandomOverSampler, SMOTE


class PRS_Dataset(Dataset):

    """ 
    Loads features and tatget. take = n -> take *first* n entries for dataset (if train = True);
    take *last* n entries for dataset (if train = False) 
    """

    def __init__(self, x_path, y_path, mode, train_index, test_index, val_index, imbalance = None):

        # print(f"### Mode: {mode} ###")
        y = np.loadtxt(y_path, delimiter=',', dtype=np.float32)
        # print(f"Overall target shape: {y.shape}")
        x = np.loadtxt(x_path, delimiter=',', dtype=np.float32)

        if mode == 'train':
            x = x[train_index]
            y = y[train_index]
            if imbalance == 'ROS':
                print(f"Y before resampling {y.shape}")
                # resampling
                ros = RandomOverSampler(random_state=5, sampling_strategy=0.35)
                x, y = ros.fit_resample(x, y)
                print(f"Y after ROS resampling {y.shape}")
            elif imbalance == 'SMOTE':
                print(f"Y before resampling {y.shape}")
                # resampling
                smote = SMOTE()
                x, y = smote.fit_resample(x, y)
                print(f"Y after SMOTE resampling {y.shape}")

        elif mode == 'test':
            x = x[test_index]
            y = y[test_index]

        elif mode == 'val':
            x = x[val_index]
            y = y[val_index]

        else:
            raise ValueError('Incorrect mode')

        self.x_data = torch.from_numpy(x).to(torch.float32)
        self.y_data = torch.from_numpy(y).to(torch.float32)
        self.y_data = self.y_data.unsqueeze(1)
        print(f"x_data {self.x_data.shape}")
        print(f"y_data {self.y_data.shape}")

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return len(self.y_data)
