import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.impute import KNNImputer
from collections import Counter
import warnings
import os


# First convert data
def cat_no_num():
    # Load the Excel file
    file_path = 'healthcare-dataset-stroke-data-noid.xlsx'
    df = pd.read_excel(file_path)
    # Step 1: Map binary nominal data
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
    df['ever_married'] = df['ever_married'].map({'No': 0, 'Yes': 1})
    df['Residence_type'] = df['Residence_type'].map({'Rural': 0, 'Urban': 1})
    # Step 2: Get unique values from 'work_type'
    unique_values = df['work_type'].unique()

    # Step 3: Manually encode the 'work_type' column
    for value in unique_values:
        # Create a new column for each unique value, and set 1 if 'work_type' is equal to the value, otherwise 0
        df[f'work_type_{value}'] = (df['work_type'] == value).astype(int)

    # Step 4: Drop the original 'work_type' column
    df = df.drop(columns=['work_type'])
    # Save the processed DataFrame to a new Excel file
    output_path = 'preprocess/stroke-data-cat-to-num.xlsx'
    df.to_excel(output_path, index=False)
    print(f"File saved as: {output_path}")


# Function to detect and show outliers using the IQR method
def show_outliers(df):
    # Columns to focus on
    columns = ['bmi', 'avg_glucose_level']
    outliers_count = {}
    # Create a figure with subplots (one for each feature)
    fig, axes = plt.subplots(1, len(columns), figsize=(15, 5))

    if len(columns) == 1:
        axes = [axes]  # To handle the case when there's only one column

    for i, column in enumerate(columns):
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)

        # Calculate IQR
        IQR = Q3 - Q1

        # Define the outlier boundaries
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Count outliers
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        outliers_count[column] = len(outliers)

        # Plot the boxplot in the corresponding subplot
        sns.boxplot(x=df[column], ax=axes[i])
        axes[i].set_title(f'Boxplot for {column}')

    # Adjust layout for better visualization
    plt.tight_layout()
    plt.savefig('images/outliers.jpg', format='jpg')
    plt.show()

    return outliers_count


# Function to detect outliers using IQR method
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]


# Function to handle outliers and fill them with the given method
def handle_outliers(df, column, method):
    outliers = detect_outliers(df, column)
    if method == 'mean':
        # Replace outliers with the mean of the column
        mean_value = df[column].mean()
        df.loc[outliers.index, column] = mean_value
    elif method == 'median':
        # Replace outliers with the median of the column
        median_value = df[column].median()
        df.loc[outliers.index, column] = median_value
    return df


def impute_outlier(df):
    # Handle BMI and Glucose outliers and fill them with the same method
    methods = ['mean', 'median']

    for method in methods:
        # Create a copy of the original dataframe to prevent modifying it multiple times
        df_filled = df.copy()

        # Handle outliers for both BMI and Glucose
        for column in ['bmi', 'avg_glucose_level']:
            df_filled = handle_outliers(df_filled, column, method)
        # Save the file with appropriate naming
        if method == 'mean':
            df_filled.to_excel(f'preprocess/bmi_glu_mean.xlsx', index=False)
        elif method == 'median':
            df_filled.to_excel(f'preprocess/bmi_glu_median.xlsx', index=False)


# Function to perform KNN imputation for smoking_status
def knn_impute_smoking(data, k, features):
    imputed_data = data.copy()

    # Replace 'Unknown' in 'smoking_status' with NaN
    imputed_data['smoking_status'] = imputed_data['smoking_status'].replace('Unknown', np.nan)

    # Apply KNN imputation to the selected features (excluding smoking_status)
    # We make sure that the data passed to KNN is numeric
    imputed_data[features] = imputed_data[features].apply(pd.to_numeric, errors='coerce')

    # Now, for each row where smoking_status is missing, find the k nearest neighbors
    for idx, row in imputed_data.iterrows():
        if pd.isna(row['smoking_status']):  # Only impute if smoking_status is missing
            # Get the k nearest neighbors that do not have NaN for smoking_status
            neighbors = imputed_data.dropna(subset=['smoking_status'])

            # Convert row and data into numpy arrays for proper vectorized operation
            row_values = np.array(row[features].values).reshape(1, -1)  # Ensure it's a 2D array
            data_values = np.array(neighbors[features].values)  # Ensure it's a 2D array

            # Check the shape and content of the arrays to debug
            # print(f"row_values shape: {row_values.shape}, data_values shape: {data_values.shape}")

            # Manually calculate Euclidean distances (sum of squared differences)
            squared_differences = (data_values - row_values) ** 2

            # Ensure no NaN values during distance calculation
            squared_differences = np.nan_to_num(squared_differences, nan=0.0)  # Replace NaN values with 0
            # print(f"squared_differences: {squared_differences}")

            # Cast squared_differences to float type to prevent issues with sqrt
            squared_differences = squared_differences.astype(np.float64)

            # Compute the Euclidean distance
            distances = np.sqrt(np.sum(squared_differences, axis=1))  # Compute the square root of the sum
            # print(f"distances: {distances}")

            # Sort the distances and get the k nearest neighbors
            nearest_neighbors_idx = np.argsort(distances)[:k]
            # print(f"nearest_neighbors_idx: {nearest_neighbors_idx}")

            # Get the smoking_status values of the nearest neighbors
            nearest_smoking_status = neighbors.iloc[nearest_neighbors_idx]['smoking_status']

            # Count the most frequent smoking_status value (mode) among the k nearest neighbors
            mode_smoking_status = Counter(nearest_smoking_status).most_common(1)[0][0]
            # print(f"mode_smoking_status: {mode_smoking_status}")

            # Fill the missing smoking_status with the most common value
            imputed_data.at[idx, 'smoking_status'] = mode_smoking_status

    return imputed_data


# Function to impute BMI using mean or median
def impute_bmi(data, strategy='mean'):
    imputed_data = data.copy()
    if strategy == 'mean':
        imputed_data['bmi'].fillna(imputed_data['bmi'].mean(), inplace=True)
    elif strategy == 'median':
        imputed_data['bmi'].fillna(imputed_data['bmi'].median(), inplace=True)
    return imputed_data


# Function to perform KNN imputation for BMI (mean of k nearest neighbors)
def knn_impute_bmi(imputed_data, k, features):
    # Prepare the KNN imputer
    knn_imputer = KNNImputer(n_neighbors=k)
    # Get the data for KNN (excluding the 'bmi' column)
    knn_data = imputed_data[features]
    # Apply KNN imputation and assign only the 'bmi' column back to the DataFrame
    imputed_bmi = knn_imputer.fit_transform(knn_data)
    # Now, update only the 'bmi' column in the DataFrame
    imputed_data['bmi'] = imputed_bmi[:, 0]  # Only assign the imputed 'bmi' values
    return imputed_data


# Main function to handle the full imputation task
def impute_missing_values(input_file, output_dir):
    # Read the data
    data = pd.read_excel(input_file)
    # Features for KNN
    knn_features = [
        'age', 'hypertension', 'heart_disease', 'avg_glucose_level',
        'work_type_Private', 'work_type_Self-employed', 'work_type_Govt_job',
        'work_type_children', 'work_type_Never_worked'
    ]

    # Updated strategies to use KNN for smoking_status in all cases
    strategies = [
        {'bmi': 'mean', 'smoke': 'knn'},
        {'bmi': 'median', 'smoke': 'knn'},
        {'bmi': 'knn', 'smoke': 'knn'}
    ]

    # Loop over each strategy and save files accordingly
    for strategy in strategies:
        for k in range(2, 11):  # K values from 2 to 10
            imputed_data = data.copy()

            # Impute BMI
            if strategy['bmi'] == 'mean':
                imputed_data = impute_bmi(imputed_data, strategy='mean')
            elif strategy['bmi'] == 'median':
                imputed_data = impute_bmi(imputed_data, strategy='median')
            elif strategy['bmi'] == 'knn':
                imputed_data = knn_impute_bmi(imputed_data, k, knn_features)

            # Impute smoking_status always using KNN
            imputed_data = knn_impute_smoking(imputed_data, k, knn_features)

            # Determine output file name
            base_name = os.path.basename(input_file).split('.')[0]
            output_file = f"{base_name}_bmi_{strategy['bmi']}_smoke_k{k}.xlsx"
            output_path = os.path.join(output_dir, output_file)

            # Save to Excel
            imputed_data.to_excel(output_path, index=False)
            print(f"Saved: {output_path}")


def normalize_and_encode(df):
    # Manually normalize age, avg_glucose_level, and bmi using (value - min) / (max - min)
    for column in ['age', 'avg_glucose_level', 'bmi']:
        min_value = df[column].min()
        max_value = df[column].max()
        df[column] = (df[column] - min_value) / (max_value - min_value)

    # Manually encode the smoking_status column
    unique_values = df['smoking_status'].unique()

    for value in unique_values:
        # Create a new column for each unique value, setting 1 if 'smoking_status' is equal to the value, otherwise 0
        df[f'smoking_status{value}'] = (df['smoking_status'] == value).astype(int)

    # Drop the original 'smoking_status' column
    df = df.drop(columns=['smoking_status'])

    return df


def process_files(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through each file in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.xlsx'):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)

            # Load the Excel file
            df = pd.read_excel(input_path)

            # Normalize the features and one-hot encode smoking_status
            cleaned_df = normalize_and_encode(df)

            # Save the cleaned DataFrame to the output folder
            cleaned_df.to_excel(output_path, index=False)
            print(f'Processed {file_name} and saved to {output_path}')



def main():
    warnings.filterwarnings('ignore')
    # cat_no_num()
    # Load your dataset
    # file_path = 'preprocess/stroke-data-cat-to-num.xlsx'
    file_path = 'healthcare-dataset-stroke-data-noid.xlsx'
    df = pd.read_excel(file_path)
    # Get the count of outliers for each numerical feature
    # outliers = show_outliers(df)

    # Print the outliers count for each feature
    # for feature, count in outliers.items():
    #     print(f'Feature: {feature}, Outliers: {count}')
    # Execute the outlier handling and imputation process

    # impute_outlier(df)
    # input_files = ['preprocess/bmi_glu_median.xlsx', 'preprocess/bmi_glu_mean.xlsx']
    # output_directory = 'preprocess/impute_missing_values/'
    #
    # for input_file in input_files:
    #     impute_missing_values(input_file, output_directory)

    # input_folder = 'preprocess/impute_missing_values'
    # output_folder = 'preprocess/impute_missing_values/cleaned'
    #
    # # Process all files in the input folder
    # process_files(input_folder, output_folder)


if __name__ == "__main__":
    main()
