import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


# Function to get null counts in each feature, treating 'Unknown' as null in smoking_status
def print_null_counts(df):
    df_copy = df.copy()

    # Treat 'Unknown' in 'smoking_status' as NaN
    if 'smoking_status' in df_copy.columns:
        df_copy['smoking_status'] = df_copy['smoking_status'].replace('Unknown', pd.NA)

    # Count nulls
    null_counts = df_copy.isna().sum()

    # Print results
    for col, count in null_counts.items():
        print(f"{col}: {count} null values")


def summarize_features(data):
    """Summarize features by providing count, max, min, and median for numerical features,
       and value counts for categorical features."""
    summary = {}

    for column in data.columns:
        if data[column].dtype in ['float64', 'int64']:  # Numerical features
            summary[column] = {
                'count': data[column].count(),
                'max': data[column].max(),
                'min': data[column].min(),
                'median': data[column].median()
            }
        else:  # Categorical features
            summary[column] = data[column].value_counts().to_dict()

    # Print summary for each feature
    for feature, stats in summary.items():
        print(f"Feature: {feature}")
        if isinstance(stats, dict) and 'count' in stats:
            print(f"  Count: {stats['count']}, Max: {stats['max']}, Min: {stats['min']}, Median: {stats['median']}")
        else:
            print("  Value Counts:")
            for value, count in stats.items():
                print(f"    {value}: {count}")


# Function to plot and save bar charts for all categorical features
def plot_and_save_categorical_bars(data):
    for column in data.select_dtypes(include=['object', 'category']).columns:
        value_counts = data[column].value_counts()  # Get counts for each unique value

        # Plotting
        plt.figure(figsize=(8, 6))
        value_counts.plot(kind='bar', color='skyblue')
        plt.title(f'Count of Values in Feature: {column}')
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.xticks(rotation=45)

        # Save plot as JPG
        plt.savefig(f'images/{column}_value_counts.jpg', format='jpg')
        plt.close()  # Close the figure to free memory


# Function to plot a heatmap of correlations in the dataset
def heatmap_label_encode(df):
    df_copy = df.copy()
    # Treat 'Unknown' in 'smoking_status' as NaN
    if 'smoking_status' in df_copy.columns:
        df_copy['smoking_status'] = df_copy['smoking_status'].replace('Unknown', pd.NA)

    for col in df_copy.select_dtypes(include=['object']).columns:
        df_copy[col] = df_copy[col].astype('category').cat.codes

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_copy.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

    # Save the figure
    plt.title("Feature Correlation Heatmap")
    plt.savefig('images/feature_correlation_heatmap.jpg', format='jpg')
    plt.show()


def heatmap_one_hot_encode(df):
    if 'smoking_status' in df.columns:
        df['smoking_status'] = df['smoking_status'].replace('Unknown', pd.NA)

    # One-hot encode categorical variables
    df_numeric = pd.get_dummies(df, drop_first=True)  # Avoid dummy variable trap

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

    # Save the figure
    plt.title("Feature Correlation Heatmap")
    plt.savefig('images/feature_correlation_heatmap_one_hot.jpg', format='jpg')
    plt.show()


# Function to plot and save bar charts for specified features
def plot_and_save_bar_charts(data):
    # List of features to plot
    features = ['hypertension', 'heart_disease', 'stroke']

    # Loop through each feature and create a bar chart
    for feature in features:
        if feature in data.columns:
            # Count the occurrences of each value (0 and 1)
            value_counts = data[feature].value_counts()

            # Plotting
            plt.figure(figsize=(6, 4))
            value_counts.plot(kind='bar', color='skyblue')
            plt.title(f'Count of Values in Feature: {feature}')
            plt.xlabel('Value')
            plt.ylabel('Count')
            plt.xticks(rotation=0)

            # Save plot as a .jpg file
            plt.savefig(f'{feature}_value_counts.jpg', format='jpg', dpi=300)
            plt.close()  # Close the plot to free up memory


# Function to remove outlier 'gender' = 'Other'
def remove_other_gender(filepath):
    # Load the data
    data = pd.read_excel(filepath)

    # Filter out rows where gender is 'Other'
    data_cleaned = data[data['gender'] != 'Other'].copy()

    # Save the cleaned data back to the original file
    data_cleaned.to_excel(filepath, index=False)
    print(f"Records with 'gender=Other' removed and data saved to {filepath}")


# scatter plot of data
def scatterPlot(data):
    dff = data.copy()
    sns.pairplot(data=dff, hue='stroke', kind='scatter', palette='bright')
    plt.savefig('images/scatterPlot.jpg', format='jpg')
    plt.show()


# pie chart of stroke 0 and 1
def piechart(df):
    outcome_counts = df["stroke"].value_counts()
    outcome_counts.plot(kind="pie", autopct="%1.1f%%", startangle=90)
    plt.title("Distribution of Stroke")
    plt.ylabel("")
    plt.savefig('images/piechart_stroke.jpg', format='jpg')
    plt.show()


# data feature description
def describe(df):
    summary = df.describe()
    fig, ax = plt.subplots(figsize=(8, 4))
    table_data = []
    for col in summary.columns:
        table_data.append([col] + list(summary[col].values))

    table = ax.table(cellText=table_data,
                     colLabels=['Statistic'] + ['count', 'mean', 'std', 'min', '25%', '50%', '70%', 'max'],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    ax.axis('off')
    plt.savefig('images/description.jpg', format='jpg')
    plt.show()


# bar charts for data
def all_bar_charts(df):
    # Fix: Replace 'Unknown' in smoking_status with NaN (standardized format)
    df['smoking_status'] = df['smoking_status'].replace('Unknown', None)

    # Select categorical columns + binary columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    binary_columns = ['hypertension', 'heart_disease', 'stroke']

    # Combine categorical + binary features
    all_categorical = categorical_columns + binary_columns

    # Set up subplots
    num_cols = len(all_categorical)
    fig, axes = plt.subplots(nrows=(num_cols // 3) + 1, ncols=3, figsize=(15, 5 * ((num_cols // 3) + 1)))
    axes = axes.flatten()  # Flatten axes for easy iteration

    # Plot bar charts for all categorical & binary columns
    for i, col in enumerate(all_categorical):
        ax = axes[i]
        sns.countplot(x=df[col], ax=ax, palette='coolwarm')

        # Add counts on bars
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

        ax.set_title(col, fontsize=14)
        ax.set_ylabel("Count")

    # Remove extra empty subplots (if any)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig('images/categorical_features_bar_charts.jpg', format='jpg')
    plt.show()


# Function to plot the distribution of avg_glucose_level, bmi, and age for stroke vs. non-stroke patients
def plot_distributions(df):
    # Features to visualize
    numerical_features = ['avg_glucose_level', 'bmi', 'age']

    # Set up subplots
    fig, axes = plt.subplots(nrows=1, ncols=len(numerical_features), figsize=(18, 5))

    for i, feature in enumerate(numerical_features):
        ax = axes[i]
        sns.kdeplot(df[df['stroke'] == 0][feature], label="No Stroke (0)", shade=True, color='blue', ax=ax)
        sns.kdeplot(df[df['stroke'] == 1][feature], label="Stroke (1)", shade=True, color='red', ax=ax)

        ax.set_title(f"Distribution of {feature}", fontsize=14)
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        ax.legend()

    plt.tight_layout()
    plt.savefig('images/stroke_feature_distributions.jpg', format='jpg')
    plt.show()


# Function to plot a pairwise scatter plot matrix for avg_glucose_level, bmi, and age
def plot_pairwise_scatter_numerical(df):
    # Features to visualize
    features = ['avg_glucose_level', 'bmi', 'age']

    # Create pairwise scatter plot matrix
    sns.pairplot(df[features], kind='scatter', diag_kind='kde', plot_kws={'alpha': 0.5})

    # Display plot
    plt.suptitle('Scatter Plot', fontsize=16)
    plt.savefig("images/scatter_plots_for_numerical_features.jpg", format='jpg')
    plt.show()


# Main function
def main():
    # Ignore all warnings
    warnings.filterwarnings('ignore')
    file_path = 'healthcare-dataset-stroke-data-noid.xlsx'
    df = pd.read_excel(file_path)
    # Print counts of stroke = 0 and 1
    # stroke_counts = df['stroke'].value_counts()
    # print(stroke_counts)

    # Handle missing values in 'bmi' (if any)
    # df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
    # plot_pairwise_scatter_numerical(df)


# Run the main function with your file path
if __name__ == "__main__":
    main()
