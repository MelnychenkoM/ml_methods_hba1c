import argparse
import os
import sys
import pandas as pd
from tqdm.auto import tqdm
from utils.curvefit import SpectraFit
from dataset import DatasetSpectra
import matplotlib.pyplot as plt

START = 1000
END = 1720
NORMALIZATION = 'amide'

def main(data_path, domain_path, peaks_path, output_dir, fit_sample):
    data_path = os.path.expanduser(data_path)
    domain_path = os.path.expanduser(domain_path)
    peaks_path = os.path.expanduser(peaks_path)
    output_dir = os.path.expanduser(output_dir)

    peaks = pd.read_csv(peaks_path).to_numpy().reshape(-1)

    dataset = DatasetSpectra(data_path, domain_path)
    dataset.baseline_corr()
    dataset.normalization(kind=NORMALIZATION)
    dataset.select_region(START, END)

    x_values = dataset.get_wavenumbers()

    model = SpectraFit()

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    metadata_path = os.path.join(output_dir, 'metadata.txt')
    with open(metadata_path, 'w') as metadata_file:
        metadata_file.write(f"Normalization kind: {NORMALIZATION}\n")
        metadata_file.write(f"Selected region: {START} to {END}\n")

    if fit_sample:
            spectra, target = dataset[fit_sample]
            model.fit(x_values, spectra, peaks)
            fig, ax = model.plot_fit()
            fig.show()
            sys.exit(1)

    with tqdm(total=len(dataset)) as pbar:
        for i in range(len(dataset)):
            spectra, target = dataset[i]

            model.fit(x_values, spectra, peaks)

            filename = f"{i}_fit_{target}.csv"
            output_path = os.path.join(output_dir, filename)
            model.params.to_csv(output_path)

            with open(metadata_path, 'a') as metadata_file:
                metadata_file.write(f"{i} r2: {model.r2:.3f} dis: {model.discrepancy:.3e}\n")

            pbar.update(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process spectral data.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the dataset CSV file")
    parser.add_argument('--domain_path', type=str, required=True, help="Path to the domain CSV file")
    parser.add_argument('--peaks_path', type=str, required=True, help="Path to the peaks CSV file")
    parser.add_argument('--output_dir', type=str, default='~/', help="Directory to save fit info")
    parser.add_argument('--fit_sample', type=int, default=None, help="Number of the sample to fit")
    
    args = parser.parse_args()
    main(args.data_path, args.domain_path, args.peaks_path, args.output_dir, args.fit_sample)