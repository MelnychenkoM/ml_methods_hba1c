import argparse
import os
import sys
import pandas as pd
from tqdm.auto import tqdm
from utils.curvefit import SpectraFit
from dataset import DatasetSpectra
import matplotlib.pyplot as plt

START = 1000
END = 1715
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

    if fit_sample:
        spectra, hba1c, age = dataset[fit_sample]
        model.fit(x_values, spectra, peaks)
        print(f"HbA1c: {hba1c}, Age: {int(age)}")
        model.plot_fit()
        plt.show()
        sys.exit(0)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    metadata_path = os.path.join(output_dir, 'metadata.txt')
    with open(metadata_path, 'w') as metadata_file:
        metadata_file.write(f"Normalization: {NORMALIZATION}\n")
        metadata_file.write(f"Selected region: {START} to {END}\n")
        metadata_file.write(f"Peaks: {peaks}\n")

    fit_info_path = os.path.join(output_dir, 'fit_info.txt')

    with open(fit_info_path, 'a') as fit_info_file:
        fit_info_file.write(f"Index\tr2\tDiscrepancy\tHbA1c\tAge\n")

    with tqdm(total=len(dataset)) as pbar:
        for i in range(len(dataset)):
            spectra, hba1c, age = dataset[i]

            try:
                model.fit(x_values, spectra, peaks)
                filename = f"{i}_fit_{hba1c}_{int(age)}.csv"
                output_path = os.path.join(output_dir, filename)
                model.params.to_csv(output_path)

                with open(fit_info_path, 'a') as fit_info_file:
                    fit_info_file.write(f"{i}\t{model.r2:.5f}\t{model.discrepancy:.5e}\t{hba1c}\t{age}\n")

            except Exception as e:
                message = f"Error at index {i}: {e}"
                with open(fit_info_path, 'a') as fit_info_file:
                    fit_info_file.write(f"{i}\tNone\tNone\tNone\tNone\n")
                print(message)

            pbar.update(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process spectral data.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the dataset CSV file")
    parser.add_argument('--domain_path', type=str, required=True, help="Path to the domain CSV file")
    parser.add_argument('--peaks_path', type=str, required=True, help="Path to the peaks CSV file")
    parser.add_argument('--output_dir', type=str, default='~/data/fits/', help="Directory to save fit info")
    parser.add_argument('--fit_sample', type=int, default=None, help="Number of the sample to fit")
    
    args = parser.parse_args()
    main(args.data_path, args.domain_path, args.peaks_path, args.output_dir, args.fit_sample)