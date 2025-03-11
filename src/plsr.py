import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

DEFAULT_COLOR = '#1f77b4'

class PLSRComponents:
    def __init__(self, **kwargs) -> None:
        self.plsr_params = kwargs
        
    def fit(self, X, y, ncomp, cv=5, threshold=0.05):
        self._ncomp = ncomp
        self._x_train = X
        self._y_train = y

        self.r2s = []
        self.rmses = []
        self.y_cvs = []
        self.q2s = []
        self.press_l = []
        self.num_comp = None
        
        ress_l_minus_1 = np.sum((y - np.mean(y)) ** 2)
        
        for num_components in range(1, ncomp + 1):
            pls = PLSRegression(n_components=num_components, **self.plsr_params)

            y_cv = cross_val_predict(pls, X, y, cv=cv)
            r2 = r2_score(y, y_cv)
            rmse = np.sqrt(mean_squared_error(y, y_cv))

            press = np.sum((y - y_cv) ** 2)
            self.press_l.append(press)
            q2 = 1 - (press / ress_l_minus_1)
            self.q2s.append(q2)

            ress_l_minus_1 = press

            self.r2s.append(r2)
            self.rmses.append(rmse)
            self.y_cvs.append(y_cv)

    
            if q2 < threshold and self.num_comp is None:
                self.num_comp = num_components

        r2 = self.r2s[self.num_comp]
        rmse = self.rmses[self.num_comp]

        pls = PLSRegression(n_components=self.num_comp, **self.plsr_params)
        pls.fit(X, y)
        
        self._fitted_model = pls
    
        return self.num_comp
    
    def plot_number_components(self):
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        y_cv = self.y_cvs[self.num_comp]

        sns.lineplot(x=np.arange(1, self._ncomp + 1), y=self.r2s, ax=axs[0, 0], marker='o', 
                    color='k', markeredgecolor='k', markerfacecolor=DEFAULT_COLOR, markersize=6)
        axs[0, 0].set_xlabel("Number of components")
        axs[0, 0].set_ylabel("R²")
        axs[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True))

        sns.lineplot(x=np.arange(1, self._ncomp + 1), y=self.rmses, ax=axs[0, 1], marker='o', 
                    color='k', markeredgecolor='k', markerfacecolor=DEFAULT_COLOR, markersize=6)
        axs[0, 1].set_xlabel("Number of components")
        axs[0, 1].set_ylabel("RMSECV")
        axs[0, 1].xaxis.set_major_locator(MaxNLocator(integer=True))

        sns.scatterplot(x= self._y_train, y=y_cv, ax=axs[1, 0], edgecolor='k', s=25)
        axs[1, 0].plot(self._y_train, self._y_train, color='k', linestyle='--', linewidth=0.9)
        axs[1, 0].set_xlabel("HbA1c% measured")
        axs[1, 0].set_ylabel("HbA1c% predicted")

        residuals = y_cv - self._y_train # predicted - measured
        sns.scatterplot(x=self._y_train, y=residuals, ax=axs[1, 1], edgecolor='k', s=25)
        axs[1, 1].axhline(y=0, color='k', linestyle='--', linewidth=0.9)
        axs[1, 1].set_xlabel("HbA1c%")
        axs[1, 1].set_ylabel("Residuals")

        return fig, axs

    def evaluate(self, X_test, y_test):
        y_pred = self._fitted_model.predict(X_test)
        fig, axs = validate_plot(y_test, y_pred)
        return fig, axs
    
    def get_fitted_model(self):
        return self._fitted_model
    
    def predict(self, X_test):
        pred = self._fitted_model.predict(X_test)
        return pred
    
def validate_plot(y_true, y_predicted, params={}):
    """
    Plot y true against y predicted
    """
    fig, axs = plt.subplots(figsize=(6, 4))
    
    r2 = r2_score(y_true, y_predicted)
    r = pearsonr(y_true, y_predicted)[0]
    mse = mean_squared_error(y_true, y_predicted)
    mae = (np.abs(y_true - y_predicted)).mean()
    rmse = np.sqrt(mse)

    sns.scatterplot(x=y_true, y=y_predicted, ax=axs, facecolor=DEFAULT_COLOR, edgecolor='k', s=25)
    
    axs.plot(y_true, y_true, color='k', linewidth=0.9)

    axs.set_xlabel("HbA1c% measured")
    axs.set_ylabel("HbA1c% predicted")

    y_min = min(y_true)
    y_max = max(y_predicted) + 1

    if params:
        x_coord = params['x']
        y_coord = params['y']
    
    plt.text(x_coord, y_coord, 
             f"R² = {r2:.3f}\nR = {r:.3f}\nMAE: {mae:.3f}\nRMSE = {rmse:.3f}\nMSE = {mse:.3f}", 
             verticalalignment='top')    

    return fig, axs


def read_spa(filepath):
    '''
    Reads a *.spa file and returns spectra and wavenumbers.
    
    Parameters
    ----------
    filepath : str
        The path to the *.spa file to read.
    
    Returns
    -------
    np.ndarray
        A 2D array where the first column is wavenumbers and the second is spectra.
    '''
    with open(filepath, 'rb') as file:
        file.seek(564)
        spectrum_points = np.fromfile(file, np.int32, 1)[0]
        
        file.seek(30)
        spectra_titles = np.fromfile(file, np.uint8, 255)
        spectra_titles = ''.join([chr(x) for x in spectra_titles if x != 0])

        file.seek(576)
        max_wavenumber = np.fromfile(file, np.single, 1)[0]
        min_wavenumber = np.fromfile(file, np.single, 1)[0]
        
        wavenumbers = np.flip(np.linspace(min_wavenumber, max_wavenumber, spectrum_points))

        file.seek(288)
        flag = 0
        while flag != 3:
            flag = np.fromfile(file, np.uint16, 1)

        data_position = np.fromfile(file, np.uint16, 1)
        file.seek(data_position[0])

        spectra = np.fromfile(file, np.single, spectrum_points)
        
    return np.stack((wavenumbers, spectra), axis=1)