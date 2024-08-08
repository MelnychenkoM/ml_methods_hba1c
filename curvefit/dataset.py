import pandas as pd
import numpy as np
from boxsers.preprocessing import rubberband_baseline_cor
from scipy.signal import savgol_filter

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

class DatasetSpectra:
    def __init__(self, file_path, domain_path):
        try:
            self.data = pd.read_csv(file_path)
            self.spectra = self.data.iloc[:, :-2].to_numpy()
            self.n_samples = self.spectra.shape[0]
            self.n_features = self.spectra.shape[1]
            self.wavenumbers = pd.read_csv(domain_path).to_numpy().reshape(-1)
            self.hba1c = self.data.iloc[:, -2].to_numpy()
            self.age = self.data.iloc[:, -1].to_numpy()
        except Exception as e:
            print(f"The following error occured while loading the dataset: {e}")

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, key):
        return self.spectra[key], self.hba1c[key], self.age[key]
    
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
        """
        Rubberband baseline correction
        """
        self.spectra = np.apply_along_axis(lambda row: rubberband_baseline_cor(row).squeeze(), 1, self.spectra)
        return None
    
    def normalization(self, kind='amide'):
        """
        Spectra normalization
        """
        if kind == 'amide':
            self.spectra = self.spectra / self.spectra.max(axis=1, keepdims=True)
        elif kind == 'vector':
            norms = np.linalg.norm(self.spectra, axis=1, keepdims=True)
            self.spectra = self.spectra / norms
        elif kind == 'snv':
            means = self.spectra.mean(axis=1, keepdims=True)
            std = self.spectra.std(axis=1, keepdims=True)
            self.spectra = self.spectra - means / std
        else:
            raise ValueError("Parameter 'kind' should be either amide|vector|snv")
        return None
    
    def savgol_filter(self, **kwargs):
        """
        Savitzky-Golay filter
        """
        self.spectra = np.apply_along_axis(lambda row: savgol_filter(row, **kwargs).squeeze(), 1, self.spectra)
    
    def select_region(self, start, end):
        """
        Selects 
        """
        mask = (self.wavenumbers >= start) & (self.wavenumbers <= end)
        self.spectra = self.spectra[:, mask]
        self.wavenumbers = self.wavenumbers[mask]
        self.n_features = self.spectra.shape[1]
    
    def select_max_abs(self, absorbance):
        mask = self.spectra.max(axis=1) >= absorbance
        self.spectra = self.spectra[mask, :]
        self.hba1c = self.hba1c[mask]
        self.age = self.age[mask]
        self.n_samples = self.spectra.shape[0]

    def plot_spectra(self, target='hba1c'):
        """
        Plot spectra with colors corresponding to target values.
        Arguments:
            target - the target value to color the spectra by ('hba1c' or 'age')
        """
        if target not in ['hba1c', 'age']:
            raise ValueError("Parameter 'target' should be either 'hba1c' or 'age'")
        
        if target == 'hba1c':
            target_values = self.hba1c
        else:
            target_values = self.age

        norm = Normalize(vmin=target_values.min(), vmax=target_values.max())
        cmap = cm.viridis

        fig, ax = plt.subplots(figsize=(10, 6))

        for idx in range(self.n_samples):
            ax.plot(self.wavenumbers, self.spectra[idx], color=cmap(norm(target_values[idx])), alpha=0.7)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, label=target.capitalize())

        ax.set_xlabel('Wavenumber')
        ax.set_ylabel('Absorbance')
        plt.show()
    
    def get_spectra(self):
        return self.spectra
    
    def get_wavenumbers(self):
        return self.wavenumbers
    

