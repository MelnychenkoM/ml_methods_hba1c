import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import lightning as L

class SpectraDataset(Dataset):
    def __init__(self, spectra, target):
        self.spectra = spectra
        self.target = target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.spectra[idx], self.target[idx]
    

class SpectraDataModule(L.LightningDataModule):
    def __init__(self, data_folder, batch_size=32):
        super().__init__()
        self.data_folder = data_folder
        self.batch_size = batch_size

    def setup(self, stage=None):
        X_train = np.load(os.path.join(self.data_folder, 'X_train.npy')).astype(np.float32)
        y_train = np.load(os.path.join(self.data_folder, 'y_train.npy')).astype(np.float32)
        X_val = np.load(os.path.join(self.data_folder, 'X_val.npy')).astype(np.float32)
        y_val = np.load(os.path.join(self.data_folder, 'y_val.npy')).astype(np.float32)
        X_test = np.load(os.path.join(self.data_folder, 'X_test.npy')).astype(np.float32)
        y_test = np.load(os.path.join(self.data_folder, 'y_test.npy')).astype(np.float32)

        self.train_dataset = SpectraDataset(X_train, y_train)
        self.val_dataset = SpectraDataset(X_val, y_val)
        self.test_dataset = SpectraDataset(X_test, y_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)