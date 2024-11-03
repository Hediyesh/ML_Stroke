import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


# Function to load the data
def load_data(filepath):
    """Load data from an Excel file."""
    try:
        data = pd.read_excel(filepath)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


# Function to get null counts in each feature, treating 'Unknown' as null in smoking_status
def get_null_counts(data):
    """Get the count of null values for each feature, treating 'Unknown' in smoking_status as null."""
    data = data.copy()  # Avoid modifying the original data
    data['smoking_status'] = data['smoking_status'].replace('Unknown', pd.NA)  # Replace 'Unknown' with NA
    null_counts = data.isnull().sum()
    print("Null counts for each feature:\n", null_counts)


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
        plt.savefig(f'{column}_value_counts.jpg', format='jpg')
        plt.close()  # Close the figure to free memory


# Function to get records with age less than 1
def get_records_with_age_less_than_one(data):
    # Filter records where age is less than 1
    records = data[data['age'] < 1][['age', 'work_type']]
    count = len(records)

    # Print and return the count and filtered records
    print(f"Number of records with age less than 1: {count}")
    print("Records with age less than 1:")
    print(records)


# Function to plot a heatmap of correlations in the dataset
def plot_data_heatmap(data, output_filename="heatmap.jpg"):
    # Drop the 'id' feature if it exists
    if 'id' in data.columns:
        data = data.drop(columns=['id'])

    # Apply one-hot encoding to categorical features
    data_encoded = pd.get_dummies(data, drop_first=True)

    # Compute correlation matrix
    correlation_matrix = data_encoded.corr()

    # Plot heatmap with annotations
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Annotated Heatmap Including Categorical Features (One-Hot Encoded)")

    # Save heatmap as .jpg file
    plt.savefig(output_filename, format="jpg", dpi=300)
    plt.close()

    print(f"Heatmap saved as {output_filename}")


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


# Function to impute bmi null values with median and Unknown smoking status with knn
def impute_bmi_smoking(data):
    # Step 1: Impute BMI with Median
    data['bmi'] = data['bmi'].fillna(data['bmi'].median())

    # Step 2: Separate records with known and unknown smoking status
    known_smoking = data[data['smoking_status'] != 'Unknown']
    unknown_smoking = data[data['smoking_status'] == 'Unknown']

    # Define nominal and numeric columns
    nominal_cols = ['gender', 'ever_married', 'work_type', 'Residence_type']
    numeric_cols = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']

    # Encode nominal columns using OneHotEncoder
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    known_encoded_nominal = encoder.fit_transform(known_smoking[nominal_cols])
    unknown_encoded_nominal = encoder.transform(unknown_smoking[nominal_cols])

    # Combine numeric and encoded nominal columns for known and unknown data
    known_features = pd.concat(
        [pd.DataFrame(known_encoded_nominal), known_smoking[numeric_cols].reset_index(drop=True)],
        axis=1
    )
    unknown_features = pd.concat(
        [pd.DataFrame(unknown_encoded_nominal), unknown_smoking[numeric_cols].reset_index(drop=True)],
        axis=1
    )

    # Convert column names to strings to avoid mixed type error
    known_features.columns = known_features.columns.astype(str)
    unknown_features.columns = unknown_features.columns.astype(str)

    # Step 3: Use KNN to predict the smoking status for unknown records
    knn = KNeighborsClassifier(n_neighbors=4)
    knn.fit(known_features, known_smoking['smoking_status'])

    # Predict smoking status for unknown records
    imputed_smoking_status = knn.predict(unknown_features)
    data.loc[data['smoking_status'] == 'Unknown', 'smoking_status'] = imputed_smoking_status

    # Save the imputed data to a new Excel file
    data.to_excel('imputed_stroke_data.xlsx', index=False)
    print('Imputed data is saved.')


# Function to normal data
def normalize_data(data):
    # Map categorical features
    data['gender'] = data['gender'].map({'Male': 0, 'Female': 1}).astype(int)
    data['ever_married'] = data['ever_married'].map({'No': 0, 'Yes': 1}).astype(int)
    data['work_type'] = data['work_type'].map(
        {'Private': 1, 'Self-employed': 0.75, 'Govt_job': 0.5, 'children': 0.25, 'Never_worked': 0}).astype(float)
    data['Residence_type'] = data['Residence_type'].map({'Urban': 0, 'Rural': 1}).astype(int)
    data['smoking_status'] = data['smoking_status'].map(
        {'never smoked': 0, 'formerly smoked': 0.5, 'smokes': 1}).astype(float)

    # Apply Z-index normalization to 'age', 'avg_glucose_level', and 'bmi'
    for feature in ['age', 'avg_glucose_level', 'bmi']:
        # Rescale to [0, 1] range
        data[feature] = (data[feature] - data[feature].min()) / (data[feature].max() - data[feature].min())

    # Save to a new file
    data.to_excel("normalized_stroke_data.xlsx", index=False)
    print("Normalized data saved to 'normalized_stroke_data.xlsx'")


# Function to split test and train 30% and 70%
def split_data_test_train(data, class_column='stroke'):
    # Count the number of samples for each class
    class_counts = data[class_column].value_counts()
    print(f"Class distribution:\n{class_counts}\n")

    # Calculate 70% and 30% split for each class
    for class_value, count in class_counts.items():
        train_count = int(count * 0.7)
        test_count = count - train_count
        print(f"Class {class_value}: Total = {count}, Train = {train_count}, Test = {test_count}")

    # Separate each class and split 70% for train and 30% for test
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()

    for class_value in class_counts.index:
        class_data = data[data[class_column] == class_value]

        # Perform the 70-30 split for the current class
        class_train, class_test = train_test_split(class_data, test_size=0.3, random_state=42)

        # Append to train and test datasets
        train_data = pd.concat([train_data, class_train], ignore_index=True)
        test_data = pd.concat([test_data, class_test], ignore_index=True)

    print("\nData has been split into training and testing sets.")
    print(f"Training set size: {len(train_data)}, Testing set size: {len(test_data)}")

    train_data.to_excel("train_data.xlsx", index=False)
    test_data.to_excel("test_data.xlsx", index=False)
    print("Training and testing sets saved.")


# General function to train and evaluate a model
def evaluate_model(model, X_train, X_test, y_train, y_test, show_rules=True):
    # Train the model directly
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Print results
    print(f"\nModel: {model.__class__.__name__}")
    print("Accuracy:", accuracy)
    print("Recall:", recall)
    print("Precision:", precision)
    print("F1 Score:", f1)

    # Show rules if applicable
    if show_rules and isinstance(model, DecisionTreeClassifier):
        rules = export_text(model, feature_names=X_train.columns.to_list())
        print("\nDecision Tree Rules:\n", rules)
        print("\nTotal number of rules:", rules.count('\n'))
        plot_tree(model, feature_names=X_train.columns, filled=True)
        plt.show()

    elif show_rules and isinstance(model, RandomForestClassifier):
        for idx, estimator in enumerate(model.estimators_[:3]):  # Show rules for the first 3 trees
            print(f"\nRules for Tree {idx + 1} in Random Forest:")
            rules = export_text(estimator, feature_names=X_train.columns.to_list())
            print(rules)
            print("\nTotal number of rules:", rules.count('\n'))
            plt.figure(figsize=(15, 10))
            plot_tree(estimator, feature_names=X_train.columns, filled=True)
            plt.show()

    return accuracy, recall, precision, f1


# Show just a few trees and rules for classifiers
def visualize_some_rules_from_boosting(model, X_train, num_trees=3):
    print(f"\nVisualizing rules for the first {num_trees} trees in {model.__class__.__name__}:\n")

    # For AdaBoost
    if hasattr(model, "estimators_"):
        for idx in range(min(num_trees, len(model.estimators_))):
            tree = model.estimators_[idx]
            if isinstance(tree, (list, tuple)):  # For AdaBoost which uses nested lists
                tree = tree[0]
            print(f"\nRules for Tree {idx + 1}:\n")
            print(export_text(tree, feature_names=X_train.columns.tolist()))
            print("\n--- End of Tree ---\n")
            plt.figure(figsize=(15, 10))
            plot_tree(tree, feature_names=X_train.columns, filled=True)
            plt.show()

    # For XGBoost, access individual trees through get_booster
    elif isinstance(model, XGBClassifier):
        booster = model.get_booster()
        for idx, tree_text in enumerate(booster.get_dump(dump_format="text")[:num_trees]):
            print(f"\nRules for Tree {idx + 1} in XGBoost:\n")
            print(tree_text)
            print("\n--- End of Tree ---\n")


# Total number of rules for classifiers
def total_number_of_rules(model):
    if hasattr(model, "estimators_"):
        total_rules = sum(estimator.tree_.node_count for estimator in model.estimators_)
        print(f"Total number of rules in {model.__class__.__name__}: {total_rules}")
    elif isinstance(model, XGBClassifier):
        booster = model.get_booster()
        total_rules = sum(tree.count("\n") for tree in booster.get_dump(dump_format="text"))
        print(f"Total number of rules in XGBoost: {total_rules}")


# Cart and C4.5 implementations
def decision_tree_model(X_train, X_test, y_train, y_test):
    dt_model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
    evaluate_model(dt_model, X_train, X_test, y_train, y_test)


# Randomforest implementation
def random_forest_model(X_train, X_test, y_train, y_test):
    rf_model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    evaluate_model(rf_model, X_train, X_test, y_train, y_test)


# Adaboost implementation
def adaboost_model(X_train, X_test, y_train, y_test):
    ada_model = AdaBoostClassifier(n_estimators=50, random_state=42)
    evaluate_model(ada_model, X_train, X_test, y_train, y_test)


# XGboost implementation
def xgboost_model(X_train, X_test, y_train, y_test):
    xgb_model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
    evaluate_model(xgb_model, X_train, X_test, y_train, y_test)


# Main function
def main():
    filepath = 'normalized_stroke_data.xlsx'
    # Load data
    data = load_data(filepath)
    if data is None:
        return

    # read test and train
    train_data = pd.read_excel('train_data.xlsx')
    test_data = pd.read_excel('test_data.xlsx')

    # separate x and y in test and train data
    X_train = train_data.drop(columns=['stroke'])
    y_train = train_data['stroke']
    X_test = test_data.drop(columns=['stroke'])
    y_test = test_data['stroke']

    # Decision Tree
    # decision_tree_model(X_train, X_test, y_train, y_test)

    # Random Forest
    # random_forest_model(X_train, X_test, y_train, y_test)

    # AdaBoost
    # ada_model = AdaBoostClassifier(n_estimators=50, random_state=42)
    # evaluate_model(ada_model, X_train, X_test, y_train, y_test)
    # visualize_some_rules_from_boosting(ada_model, X_train)
    # total_number_of_rules(ada_model)

    # XGBoost
    # xgb_model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
    # evaluate_model(xgb_model, X_train, X_test, y_train, y_test)
    # visualize_some_rules_from_boosting(xgb_model, X_train)
    # total_number_of_rules(xgb_model)


# Run the main function with your file path
if __name__ == "__main__":
    main()
