import pandas as pd
import numpy as np
from boxsers.preprocessing import rubberband_baseline_cor
from scipy.signal import savgol_filter


class DatasetSpectra:
    def __init__(self, file_path, domain_path, target='hba1c'):
        self.data = pd.read_csv(file_path)
        self.spectra = self.data.iloc[:, :-2].to_numpy()
        self.n_samples = self.spectra.shape[0]
        self.n_features = self.spectra.shape[1]
        self.wavenumbers = pd.read_csv(domain_path).to_numpy().reshape(-1)

        if target == 'hba1c':
            self.target = self.data.iloc[:, -2].to_numpy()
        elif target == 'age':
            self.target = self.data.iloc[:, -1].to_numpy()
        else:
            raise ValueError("target must be == 'hba1c' or 'age'")

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, key):
        return self.spectra[key], self.target[key]
    
    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index < len(self):
            item = self[self.current_index]
            self.current_index += 1
            return item
        else:
            raise StopIteration

    def baseline_corr(self):
        self.spectra = np.apply_along_axis(lambda row: rubberband_baseline_cor(row).squeeze(), 1, self.spectra)
        return None
    
    def normalization(self, kind='amide'):
        if kind == 'amide':
            self.spectra = self.spectra / self.spectra.max(axis=1, keepdims=True)
        elif kind == 'vector':
            norms = np.linalg.norm(self.spectra, axis=1, keepdims=True)
            self.spectra = self.spectra / norms
        else:
            raise ValueError("Parameter 'kind' should be either 'amide' or 'vector'")
        return None
    
    def savgol_filter(self, **kwargs):
        self.spectra = np.apply_along_axis(lambda row: savgol_filter(row, **kwargs).squeeze(), 1, self.spectra)
    
    def select_region(self, start, end):
        mask = (self.wavenumbers >= start) & (self.wavenumbers <= end)
        self.spectra = self.spectra[:, mask]
        self.wavenumbers = self.wavenumbers[mask]
        return None
    
    def get_targets(self):
        return self.target
    
    def get_spectra(self):
        return self.spectra
    
    def get_wavenumbers(self):
        return self.wavenumbers
    
