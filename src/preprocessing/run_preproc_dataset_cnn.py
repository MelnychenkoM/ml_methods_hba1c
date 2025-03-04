import os
import argparse
import numpy as np
from dataset import DatasetSpectra
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split


def main(file_path, domain_path, max_hba1c, region, test_size, random_state):
    dataset = DatasetSpectra(file_path, domain_path)
    dataset.select_max_hba1c(max_hba1c)
    dataset.drop_samples([287, 636])

    dataset.savgol_filter(window_length=32, polyorder=2, deriv=1)
    dataset.normalization('vector')
    dataset.select_region(region)

    X = dataset.spectra
    y = dataset.hba1c
    wavenumbers = dataset.wavenumbers

    discretizer = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='uniform', random_state=random_state)
    categories = discretizer.fit_transform(y.reshape(-1, 1))

    X_temp, X_test, y_temp, y_test, categories_temp, categories_test = train_test_split(
        X, y, categories,
        test_size=0.2,
        stratify=categories,
        random_state=random_state
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.2,
        stratify=categories_temp,
        random_state=random_state
    )

    # Save the splits
    output_dir = '../../data/train_test_cnn'
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    np.save(os.path.join(output_dir, 'wavenumbers.npy'), wavenumbers)

    print("[Dataset Preparation] Train-val-test split for CNN saved successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process spectral dataset and perform train-test split.")
    parser.add_argument('--file_path', type=str, required=True, help="Path to the main dataset CSV file.")
    parser.add_argument('--domain_path', type=str, required=True, help="Path to the domain dataset CSV file.")
    parser.add_argument('--max_hba1c', type=float, default=14, help="Maximum HbA1c value to retain in the dataset.")
    parser.add_argument('--region', type=int, nargs=2, default=[[800, 1800], [2800, 3400]], help="Spectral region to select as [start, end].")
    parser.add_argument('--test_size', type=float, default=0.2, help="Proportion of the dataset to include in the test split.")
    parser.add_argument('--random_state', type=int, default=34, help="Random state for reproducibility.")

    args = parser.parse_args()

    main(
        file_path=args.file_path,
        domain_path=args.domain_path,
        max_hba1c=args.max_hba1c,
        region=args.region,
        test_size=args.test_size,
        random_state=args.random_state
    )
