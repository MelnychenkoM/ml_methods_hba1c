import numpy as np
import pandas as pd
from jaxfit import CurveFit
import jax.numpy as jnp
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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
        self.initial_guess = None
        self.model_loaded = False

    @staticmethod
    def combined_pseudo_voigt(x, *params):
        """ 
        Voigt profile approximation by 
        linear combination of Lorenzian and Gaussian distributions. 
        ------------------------------------------------------------
        Arguments:
            params - voigt profile parameters 
            (mu, gamma, amplitude, eta)
            mu - center of the distribution,
            gamma - FWHM for both gaussian and lorentzian terms,
            amplitude - peak height,
            eta - mixing parameter of the gaussian and lorentzian term

        Returns:
            jnp array
        """
        N = len(params) // 4
        result = jnp.zeros_like(x)
        for i in range(N):
            mu, gamma, amplitude, eta = params[i*4:(i+1)*4]

            a_G = (2 / gamma) * jnp.sqrt(jnp.log(2) / jnp.pi)
            b_G = (4 * jnp.log(2)) / (gamma**2)
            gaussian_term = a_G * jnp.exp(-b_G * (x - mu)**2)

            lorentzian_term = (1 / jnp.pi) * ((gamma / 2) / ((x - mu)**2 + (gamma / 2)**2))

            result += amplitude * (eta * gaussian_term + (1 - eta) * lorentzian_term)

        return result

    @staticmethod
    def calculate_FWHM_pseudo_voigt(x, y):
        """
        Calculates FWHM of a given spectra
        ----------------------------------
        Arguments:
            x - wavenumbers
            y - absorbance
    
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

    def _create_params(self, x_values, y_values, peaks, param_dict=None):
        """
        Creates initial parameters and bounds for the fit given a list of peaks.
        ------------------------------------------------------------------------
        Arguments:
            x_values -- wavenumbers
            y_values -- absorbance
            peaks -- peak indices
            param_dict -- dictionary specifying parameter values and bounds
        
        Returns:
            lower & upper bounds and initial values
        """

        default_params = {
            "center": {"min": 5, "max": 5},
            "fwhm": {"min": 0, "max": np.inf, "value": 5},
            "amplitude": {"min": 0, "max": np.inf},
            "eta": {"min": 0, "max": 1, "value": 0.5}
        }

        if param_dict:
            for key, value in param_dict.items():
                if key in default_params:
                    default_params[key].update(value)
                else:
                    print(f"Parameter {key} is not found. Proceeding with default parameters.")

        self.initial_guess = []
        self.lower_bound = []
        self.upper_bound = []

        for peak in peaks:
            wavenumber = x_values[peak]
            amplitude = y_values[peak]

            self.initial_guess.extend([
                wavenumber,                               # center
                default_params["fwhm"]["value"],          # FWHM
                amplitude,                                # A
                default_params["eta"]["value"]            # eta (initial guess, can be modified)
            ])

            self.lower_bound.extend([
                wavenumber - default_params["center"]["min"],   # Lower bound for center
                default_params["fwhm"]["min"],                  # Lower bound for FWHM
                default_params["amplitude"]["min"],             # Lower bound for A
                default_params["eta"]["min"]                    # Lower bound for eta
            ])

            self.upper_bound.extend([
                wavenumber + default_params["center"]["max"],   # Upper bound for center
                default_params["fwhm"]["max"],                  # Upper bound for FWHM
                default_params["amplitude"]["max"],             # Upper bound for A
                default_params["eta"]["max"]                    # Upper bound for eta
            ])

        return None
    
    def load_model(self, model):
        """
        Loads the bounds and initial guess from another model, 
        disregarding any parameters provided in param_dict. 
        This function is intended for fitting spectra based on a previous fit.
        ---------------------------------------------------------------------
        Arguments:
            model - an instance of SpectraFit
        """
        if isinstance(model, SpectraFit):
            try:
                self.initial_guess = model.params_array
                self.lower_bound = model.lower_bound
                self.upper_bound = model.upper_bound
                self.model_loaded = True
            except Exception as e:
                print(f"The following error occured while loading the model: {e}")
        else:
            raise ValueError("model has to be an instance of SpectraFit")
        return None

    def fit(self, x_values, y_values, peaks, param_dict=None, **kwargs):
        """
        Curve fitting using jaxfit.
        -------------------------------------
        Arguments:
            x_values - wavenumbers
            y_values - absorbance
            peaks - peak indices
            kwargs -- keyword arguments for curve_fit
        
        Returns:
            fit parameters
        """
        self.x_values = x_values
        self.y_values = y_values
        self.peaks = [np.argmin(np.abs(self.x_values - peak)) for peak in peaks]

        if not self.model_loaded:
            self._create_params(self.x_values, 
                                self.y_values, 
                                self.peaks,
                                param_dict=param_dict
                                )

        jcf = CurveFit()

        bounds = (self.lower_bound, self.upper_bound)

        params_array, pcov = jcf.curve_fit(self.combined_pseudo_voigt,
                                    self.x_values,
                                    self.y_values,
                                    p0=self.initial_guess,
                                    bounds=bounds,
                                    **kwargs,
                                    )
        
        self.params_array = params_array
        self.pcov = pcov
        self.predicted = self.combined_pseudo_voigt(self.x_values, *params_array)
        self.discrepancy = np.sqrt(np.mean((self.predicted - self.y_values)**2))
        self.r2 = r2_score(self.y_values, self.predicted)

        areas = []
        areas_abs = []
        height = []

        y_combined = self.combined_pseudo_voigt(self.x_values, *params_array)
        total_area = np.trapz(y_combined, x=self.x_values)

        for i in range(0, len(params_array), 4):
            y_pred = self.combined_pseudo_voigt(self.x_values, *params_array[i:i+4])
            height.append(max(y_pred))
            areas.append(np.trapz(y_pred / total_area, x=self.x_values))
            areas_abs.append(np.trapz(y_pred, x=self.x_values))


        self.wavenumbers = params_array[::4]
        self.gamma = params_array[1::4]
        self.amplitude = params_array[2::4]
        self.eta = params_array[3::4]
        self.areas = np.array(areas)
        self.areas_abs = np.array(areas_abs)
        self.height = np.array(height)

        self.params = pd.DataFrame({
            "wavenumber": self.wavenumbers,
            "FWHM": self.gamma,
            "amplitude": self.amplitude,
            "eta": self.eta,
            "height": self.height,
            "area": self.areas,
            "area_absolute": areas_abs
        })

        return self.params

    def plot_fit(self, kind='fit'):
        """
        Plot resulting fit/residuals
        """
        if self.params is not None:
            if kind == 'fit':
                fig, ax = plt.subplots(figsize=(10, 6))

                ax.plot(self.x_values, self.y_values, label='Absorbance', color='blue')
                ax.plot(self.x_values, self.predicted, label='Best fit', color='black')
                # ax.scatter(self.x_values[self.peaks], self.y_values[self.peaks], label='Peaks', color='red', s=10)

                for index, row in self.params.iterrows():
                    pseudo_voigt = self.combined_pseudo_voigt(self.x_values, 
                                                            row["wavenumber"], 
                                                            row["FWHM"], 
                                                            row["amplitude"], 
                                                            row["eta"])
                    ax.plot(self.x_values, pseudo_voigt, linestyle='--', linewidth=0.6, color='k')

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
                ax.set_ylabel('Residuals')
                ax.legend()
            
            else:
                raise ValueError('Kind should be either "fit" or "residuals"')
        else:
            raise ValueError("You need to fit a model first.")
    
        return fig, ax
    
    def plot_fit_plotly(self, kind='fit'):
        """
        Plot resulting fit/residuals
        """
        if self.params is not None:
            if kind == 'fit':
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=self.x_values, y=self.y_values, mode='lines', name='Absorbance'))
                fig.add_trace(go.Scatter(x=self.x_values, y=self.predicted, mode='lines', name='Best fit', line=dict(color='black')))
                fig.add_trace(go.Scatter(
                    x=self.x_values[self.peaks],
                    y=self.y_values[self.peaks],
                    mode='markers',
                    name='Peaks',
                    marker=dict(size=8, color='red'),
                    visible='legendonly'
                ))

                fig.update_layout(title='Pseudo Voigt fit', 
                      xaxis_title='Wavenumber', 
                      yaxis_title='Absorbance', 
                      height=650, 
                      width=950
                     )


                for index, row in self.params.iterrows():
                    pseudo_voigt = self.combined_pseudo_voigt(self.x_values, 
                                                            row["wavenumber"], 
                                                            row["FWHM"], 
                                                            row["amplitude"], 
                                                            row["eta"])
                    fig.add_trace(go.Scatter(x=self.x_values, 
                                 y=pseudo_voigt, 
                                 mode='lines', 
                                 line=dict(width=1, dash='dash'),
                                 name=str(round(row['wavenumber'], 2))))

            elif kind == 'residuals':
                residual = self.y_values - self.predicted
    
                fig_residuals = go.Figure()
                fig_residuals.add_trace(go.Scatter(x=self.x_values, y=residual, mode='lines', name='Residuals', line=dict(color='green')))
                fig_residuals.update_layout(title='Pseudo Voigt fit Residuals', 
                        xaxis_title='Wavenumber', 
                        yaxis_title='Residual', 
                        height=650, 
                        width=950
                        )
            
            else:
                raise ValueError('Kind should be either "fit" or "residuals"')
        else:
            raise ValueError("You need to fit a model first.")
    
        return fig
