# Machine Learning Methods for HbA1c prediction
This repository contains code for glycated haemoglobin levels prediction using machine learing methods.

### DatasetSpectra
The code features custom class `DatasetSpectra` for spectra loading and preprocessing. Some of its methods contain `select_region`, `savgol_filter` and `plot_spectra`.
The example of usage can be found below:
```
dataset = DatasetSpectra(file_path, domain_path)
dataset.select_region([400, 4000])
fig, ax = dataset.plot_spectra()
```
The shapes required for the `DatasetSpectra` class are hardcoded for convenience. The dataset (file path parameter) is expected to have a shape of [n_features + 2, n_samples] , with the last two rows representing the age and HbA1c content of the participants. The domain should have a shape of [n_features, 1] and corresponds to the measurement domain, which is in units of 1/cm (in our case).
### Spectra fit
The class for curve fitting, which utilizes the JAXFit library, was constructed to facilitate the modeling and analysis of spectral data.
Please note that the peaks should be identified beforehand; functions for peak detection are provided in the `peaks.ipynb` notebook included in this repository.
```
spectra, hba1c, age = dataset[32]
x_values = dataset.get_wavenumbers()

params = {
    "center": {"min": 3, "max": 3},
    "fwhm": {"value": 10}
}

pmodel = SpectraFit()
pmodel.fit(x_values, spectra, peaks, params, ftol=1e-12)
fig = pmodel.plot_fit_plotly()
fig.show()
print(f"R2 : {pmodel.r2:.5f}, DIS: {pmodel.discrepancy:.3e}")
```
### Examples on Google Colab

Here are some simple examples on Colab: [Google Colab Examples](https://colab.research.google.com/drive/1qpJxO0BUHP2cI3RBplI76r7Ra5AAwCYI?usp=sharing)