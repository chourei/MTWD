import h5py as h5py
import numpy as np
import loadGDFdata as gdf
import torch
from torch.utils.data.dataset import Dataset
class EEGDataset(Dataset):
    def __init__(self,label,data):
        self.label = label
        self.data = data
    def __len__(self):
        return len(self.label)
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data[idx]
        label = self.label[idx]
        sample = (data,label)
        return sample
def index_of_ten_folds(num):
    index = np.zeros((10,1))
    numOFgroup = int(num/10)
    numOFextra = num%10
    for i in range(10):
        if i < numOFextra:
            index[i] = int(numOFgroup)+int(1)
        else:
            index[i] = int(numOFgroup)
    return index
def Subject9_i_10fold(i):
    train_data,train_label,test_data,test_label = gdf.read_trainANDtest(i)

    data = np.vstack((train_data,test_data))
    label = np.vstack((train_label,test_label))
    np.random.seed(12)
    np.random.shuffle(data)
    np.random.seed(12)
    np.random.shuffle(label)
    EEG = EEGDataset(label=label, data=data)
    return EEG