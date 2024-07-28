import numpy as np
import pandas as pd
from jaxfit import CurveFit
import jax.numpy as jnp
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

class SpectraFit:
    def __init__(self):
        self.x_values = None
        self.y_values = None
        self.peaks = None
        self.params = None
        self.pcov = None
        self.predicted = None
        self.discrepancy = None
        self.r2 = None

    @staticmethod
    def combinded_gaussian(x, *params):
        return

    @staticmethod
    def combined_pseudo_voigt(x, *params):
        """ 
        Voigt profile approximation by 
        linear combination of Lorenzian and Gaussian distributions. 
        ------------------------------------------------------------
        Arguments:
            params - voigt profile parameters 
            (mu, gamma_gaussian, gamma_lorentzian, amplitude, eta)
            mu - center of the distribution,
            gamma_gaussion - FWHM of gaussian term,
            gamma_lorentzian - FWHM of lorentzian term,
            eta - mixing parameter of the gaussian and lorentzian term

        Returns:
            jnp array
        """
        N = len(params) // 5
        result = jnp.zeros_like(x)
        for i in range(N):
            mu, gamma_gaussian, gamma_lorentzian, amplitude, eta = params[i*5:(i+1)*5]

            a_G = (2 / gamma_gaussian) * jnp.sqrt(jnp.log(2) / jnp.pi)
            b_G = (4 * jnp.log(2)) / (gamma_gaussian**2)
            gaussian_term = a_G * jnp.exp(-b_G * (x - mu)**2)

            lorentzian_term = (1 / jnp.pi) * ((gamma_lorentzian / 2) / ((x - mu)**2 + (gamma_lorentzian / 2)**2))

            result += amplitude * (eta * gaussian_term + (1 - eta) * lorentzian_term)

        return result

    @staticmethod
    def calculate_FWHM_pseudo_voigt(x, y):
        """
        Calculates FWHM of a givem spectra
        ----------------------------------
        Arguments:
            x - wavenumbers
            y - abrosbance
    
        Returns:
            FWHM
        """

        max_index = np.argmax(y)
        max_y = y[max_index]

        half_max = max_y / 2

        left_index = np.argmin(np.abs(y[:max_index] - half_max))
        right_index = max_index + np.argmin(np.abs(y[max_index:] - half_max))

        x_at_half_max_left = np.interp(half_max, y[left_index:left_index+2], x[left_index:left_index+2])
        x_at_half_max_right = np.interp(half_max, y[right_index:right_index+2], x[right_index:right_index+2])

        FWHM = x_at_half_max_right - x_at_half_max_left

        return FWHM

    @staticmethod
    def create_params_pseudo_voigt(x_values, y_values, peaks, param_dict=None):
        """
        Creates initial parameters and bounds for the fit given a list of peaks.
        ------------------------------------------------------------------------
        Arguments:
            x_values -- wavenumbers
            y_values -- absorbance
            peaks -- peak indicies
            param_dict -- dictionary specifying parameter values and bounds
        
        Returns:
            lower & upper bounds and initial values
        """

        default_params = {
            "center": {"min": 5, "max": 5},
            "fwhm_gauss": {"min": 0, "max": np.inf, "value": 5},
            "fwhm_lorentz": {"min": 0, "max": np.inf, "value": 5},
            "amplitude": {"min": 0, "max": np.inf},
            "eta": {"min": 0, "max": 1, "value": 0.5}
        }

        if param_dict:
            for key, value in param_dict.items():
                if key in default_params:
                    default_params[key].update(value)

        params = []
        lower_bound = []
        upper_bound = []

        for peak in peaks:
            wavenumber = x_values[peak]
            amplitude = y_values[peak]

            params.extend([
                wavenumber,                               # center
                default_params["fwhm_gauss"]["value"],    # FWHM Gaussian
                default_params["fwhm_lorentz"]["value"],  # FWHM Lorentzian
                amplitude,                                # A
                default_params["eta"]["value"]            # eta (initial guess, can be modified)
            ])

            lower_bound.extend([
                wavenumber - default_params["center"]["min"],   # Lower bound for center
                default_params["fwhm_gauss"]["min"],            # Lower bound for Gaussian FWHM (sigma)
                default_params["fwhm_lorentz"]["min"],          # Lower bound for Lorentzian FWHM (gamma)
                default_params["amplitude"]["min"],             # Lower bound for A
                default_params["eta"]["min"]                    # Lower bound for eta
            ])

            upper_bound.extend([
                wavenumber + default_params["center"]["max"],   # Upper bound for center
                default_params["fwhm_gauss"]["max"],            # Upper bound for Gaussian FWHM (sigma)
                default_params["fwhm_lorentz"]["max"],          # Upper bound for Lorentzian FWHM (gamma)
                default_params["amplitude"]["max"],             # Upper bound for A
                default_params["eta"]["max"]                    # Upper bound for eta
            ])

        return (lower_bound, upper_bound), params

    def fit(self, x_values, y_values, peaks, param_dict=None):
        """
        Curve fitting using jaxfit.
        -------------------------------------
        Arguments:
            x_values - wavenumers
            y_values - absorbance
            peaks - peak indicies
        
        Returns:
            fit parameters
        """
        self.x_values = x_values
        self.y_values = y_values
        self.peaks = [np.argmin(np.abs(self.x_values - peak)) for peak in peaks]

        bounds, initial_guess = self.create_params_pseudo_voigt(self.x_values, 
                                                                self.y_values, 
                                                                self.peaks,
                                                                param_dict=param_dict)

        jcf = CurveFit()

        params_array, pcov = jcf.curve_fit(self.combined_pseudo_voigt,
                                    self.x_values,
                                    self.y_values,
                                    p0=initial_guess,
                                    bounds=bounds)
        
        self.pcov = pcov
        self.predicted = self.combined_pseudo_voigt(self.x_values, *params_array)
        self.discrepancy = np.sqrt(np.mean((self.predicted - self.y_values)**2))
        self.r2 = r2_score(self.y_values, self.predicted)

        # print(f"DIS: {self.discrepancy:.3e}", f"R2: {self.r2:.5f}")
        
        areas = []
        fwhms = []
        height = []

        y_combined = self.combined_pseudo_voigt(self.x_values, *params_array)
        total_area = np.trapz(y_combined, x=self.x_values)

        for i in range(0, len(params_array), 5):
            y_pred = self.combined_pseudo_voigt(self.x_values, *params_array[i:i+5])
            height.append(max(y_pred))
            fwhms.append(self.calculate_FWHM_pseudo_voigt(self.x_values, y_pred))
            areas.append(np.trapz(y_pred / total_area, x=self.x_values))


        self.wavenumers = params_array[::5]
        self.gamma_gauss = params_array[1::5]
        self.gamma_lorentz = params_array[2::5]
        self.amplitude = params_array[3::5]
        self.eta = params_array[4::5]
        self.fwhm = np.array(fwhms)
        self.areas = np.array(areas)
        self.height = np.array(height)

        self.params = pd.DataFrame({
            "wavenumber": self.wavenumers,
            "fwhm_gauss": self.gamma_gauss,
            "fwhm_lorentz": self.gamma_lorentz,
            "amplitude": self.amplitude,
            "eta": self.eta,
            "FWHM": self.fwhm,
            "height": self.height,
            "area": self.areas,
        })

        return self.params

    def plot_fit(self, kind='fit'):
        """
        Plot resulting fit/residuals
        """
        if isinstance(self.params, pd.DataFrame):
            if kind == 'fit':
                fig, ax = plt.subplots(figsize=(10, 6))

                ax.plot(self.x_values, self.y_values, label='Absorbance', color='blue')
                ax.plot(self.x_values, self.predicted, label='Best fit', color='black')
                ax.scatter(self.x_values[self.peaks], self.y_values[self.peaks], label='Peaks', color='red', s=16)

                for index, row in self.params.iterrows():
                    pseudo_voigt = self.combined_pseudo_voigt(self.x_values, 
                                                            row["wavenumber"], 
                                                            row["fwhm_gauss"], 
                                                            row["fwhm_lorentz"], 
                                                            row["amplitude"], 
                                                            row["eta"])
                    ax.plot(self.x_values, pseudo_voigt, linestyle='--', linewidth=1)

                ax.set_title('Pseudo Voigt fit')
                ax.set_xlabel('Wavenumber')
                ax.set_ylabel('Absorbance')
                ax.legend()

            elif kind == 'residuals':
                fig, ax = plt.subplots(figsize=(10, 6))

                residual = self.y_values - self.predicted
                ax.plot(self.x_values, residual, label='Residual', color='green')

                ax.set_title('Pseudo Voigt fit Residuals')
                ax.set_xlabel('Wavenumber')
                ax.set_ylabel('Residual')
                ax.legend()
            
            else:
                raise ValueError('Kind should be either "fit" or "residuals"')
        else:
            raise ValueError("You need to fit a model first.")
    
        return fig, ax