import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def smote_then_split(input_folder):
    # Folder paths
    output_folder = 'split_test_train/smote_results'

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each file
    for file in os.listdir(input_folder):
        if file.endswith('.xlsx'):
            df = pd.read_excel(os.path.join(input_folder, file))

            # Split into features and target
            X = df.drop('stroke', axis=1)
            y = df['stroke']

            # Split into train and test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

            # Apply SMOTE only on training data
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

            # Save the resampled train and original test sets
            train_resampled = pd.concat([X_train_res, y_train_res], axis=1)
            test = pd.concat([X_test, y_test], axis=1)

            # Save to new files
            train_resampled.to_excel(os.path.join(output_folder, f"train_smote_{file}"))
            test.to_excel(os.path.join(output_folder, f"test_{file}"))


def split_then_smote(input_folder):
    # Folder paths
    output_folder = 'split_test_train/smote_balanced_results'

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each file
    for file in os.listdir(input_folder):
        if file.endswith('.xlsx'):
            df = pd.read_excel(os.path.join(input_folder, file))

            # Split into features and target
            X = df.drop('stroke', axis=1)
            y = df['stroke']

            # Separate the data by stroke values
            X_stroke1 = X[y == 1]
            y_stroke1 = y[y == 1]
            X_stroke0 = X[y == 0]
            y_stroke0 = y[y == 0]

            # Split stroke=1 and stroke=0 separately
            X_train_stroke1, X_test_stroke1, y_train_stroke1, y_test_stroke1 = train_test_split(
                X_stroke1, y_stroke1, test_size=0.3, random_state=42
            )
            X_train_stroke0, X_test_stroke0, y_train_stroke0, y_test_stroke0 = train_test_split(
                X_stroke0, y_stroke0, test_size=0.3, random_state=42
            )

            # Combine the stroke=1 and stroke=0 train/test sets
            X_train = pd.concat([X_train_stroke1, X_train_stroke0], axis=0)
            y_train = pd.concat([y_train_stroke1, y_train_stroke0], axis=0)
            X_test = pd.concat([X_test_stroke1, X_test_stroke0], axis=0)
            y_test = pd.concat([y_test_stroke1, y_test_stroke0], axis=0)

            # Apply SMOTE on training data
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

            # Save the resampled train and original test sets
            train_resampled = pd.concat([X_train_res, y_train_res], axis=1)
            test = pd.concat([X_test, y_test], axis=1)

            # Save to new files
            train_resampled.to_excel(os.path.join(output_folder, f"train_smote_balanced_{file}"))
            test.to_excel(os.path.join(output_folder, f"test_balanced_{file}"))


def make_samples(input_folder):
    # Folder paths
    output_sample_folder = 'split_test_train/sample_results'

    # Ensure the output folder exists
    os.makedirs(output_sample_folder, exist_ok=True)

    # Process each file
    for file in os.listdir(input_folder):
        if file.endswith('.xlsx'):
            df = pd.read_excel(os.path.join(input_folder, file))

            # Split into features and target
            X = df.drop('stroke', axis=1)
            y = df['stroke']

            # Separate the data by stroke values
            X_stroke1 = X[y == 1]
            y_stroke1 = y[y == 1]
            X_stroke0 = X[y == 0]
            y_stroke0 = y[y == 0]

            # Number of stroke=1 samples
            n_stroke1 = len(X_stroke1)

            # Generate 10 samples with stroke=1 and random stroke=0
            sample_train_idx = []
            sample_test_idx = []

            for _ in range(10):
                # Randomly sample stroke=0 data to match stroke=1
                stroke0_sampled = X_stroke0.sample(n=n_stroke1, random_state=42)

                # Combine stroke=1 and stroke=0 samples
                sample_train = pd.concat([X_stroke1, stroke0_sampled], axis=0)
                sample_train_idx.append(sample_train.index.tolist())

                # Split into train/test
                _, sample_test = train_test_split(sample_train, test_size=0.3, random_state=42)

                sample_test_idx.append(sample_test.index.tolist())

            # Save the index arrays
            np.save(os.path.join(output_sample_folder, f"train_indices_{file}.npy"), sample_train_idx)
            np.save(os.path.join(output_sample_folder, f"test_indices_{file}.npy"), sample_test_idx)


def main():
    input_folder = 'preprocess/impute_missing_values/cleaned'
    # Load the saved index arrays
    # train_indices = np.load('split_test_train/sample_results/train_indices_bmi_glu_mean_bmi_knn_smoke_k2.xlsx.npy', allow_pickle=True)
    # test_indices = np.load('split_test_train/sample_results/test_indices_bmi_glu_mean_bmi_knn_smoke_k2.xlsx.npy', allow_pickle=True)

    # Print or use the loaded indices
    # print(train_indices)
    # print(test_indices)

if __name__ == "__main__":
    main()
