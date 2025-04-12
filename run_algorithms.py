import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


def log_results_to_excel(file_path, algorithm, accuracy, precision, recall, f1_score, excel_file='run_algorithms.xlsx'):
    # Check if the Excel file already exists
    if os.path.exists(excel_file):
        # Load the existing Excel file and specify the engine
        existing_data = pd.read_excel(excel_file, engine='openpyxl')
    else:
        # Create an empty DataFrame with the correct headers if the file does not exist
        existing_data = pd.DataFrame(columns=['file_path', 'algorithm', 'accuracy', 'precision', 'recall', 'f1-score'])

    # Create a new DataFrame with the result to append
    new_data = pd.DataFrame({
        'file_path': [file_path],
        'algorithm': [algorithm],
        'accuracy': [accuracy],
        'precision': [precision],
        'recall': [recall],
        'f1-score': [f1_score]
    })

    # Concatenate the existing data with the new data
    combined_data = pd.concat([existing_data, new_data], ignore_index=True)

    # Save the combined data back to the Excel file using ExcelWriter in a safe way
    with pd.ExcelWriter(excel_file, engine='openpyxl', mode='w') as writer:
        combined_data.to_excel(writer, index=False)


def select_top_algorithms():
    # Load the Excel file
    file_path = "run_algorithms.xlsx"
    df = pd.read_excel(file_path)

    # Convert accuracy (and other metrics if needed) to numeric
    df['accuracy'] = df['accuracy'].apply(lambda x: eval(x.replace('/', '/1.')) if isinstance(x, str) else x)

    # Calculate mean accuracy for each file path
    mean_accuracy_by_file = df.groupby('file_path')['accuracy'].mean()

    # Find the file path with the highest mean accuracy
    top_file_path = mean_accuracy_by_file.idxmax()

    # Filter the DataFrame for this file path
    top_file_df = df[df['file_path'] == top_file_path]

    # Sort algorithms by accuracy in descending order
    top_algorithms = top_file_df.sort_values(by='accuracy', ascending=False)

    # Get the top 3 algorithms
    top_3_algorithms = top_algorithms[['algorithm', 'accuracy']].head(3)

    print("File path with the highest mean accuracy:", top_file_path)
    print("\nTop 3 algorithms for this file path:")
    print(top_3_algorithms)


def find_best_algorithm(file_path):
    data = pd.read_excel(file_path)
    # Define features (X) and target (y)
    target_column = 'stroke'
    X = data.drop(columns=[target_column])
    y = data[target_column]
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split the resampled data
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Convert back to DataFrame
    # df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    # df_resampled['stroke'] = y_resampled
    # train_indices, test_indices = train_test_split(df_resampled.index, test_size=0.2)
    # X_train, X_test = X.loc[train_indices], X.loc[test_indices]
    # y_train, y_test = y.loc[train_indices], y.loc[test_indices]
    # Standardize features if needed
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define models
    models = {
        "k-Nearest Neighbors 2": KNeighborsClassifier(n_neighbors=2),
        "k-Nearest Neighbors 3": KNeighborsClassifier(n_neighbors=3),
        "k-Nearest Neighbors 4": KNeighborsClassifier(n_neighbors=4),
        "k-Nearest Neighbors 5": KNeighborsClassifier(n_neighbors=5),
        "k-Nearest Neighbors 6": KNeighborsClassifier(n_neighbors=6),
        "k-Nearest Neighbors 7": KNeighborsClassifier(n_neighbors=7),
        "k-Nearest Neighbors 8": KNeighborsClassifier(n_neighbors=8),
        "k-Nearest Neighbors 9": KNeighborsClassifier(n_neighbors=9),
        "k-Nearest Neighbors 10": KNeighborsClassifier(n_neighbors=10),
        "Random Forest 100": RandomForestClassifier(n_estimators=100),
        "Random Forest 90": RandomForestClassifier(n_estimators=90),
        "Random Forest 80": RandomForestClassifier(n_estimators=80),
        "Random Forest 70": RandomForestClassifier(n_estimators=70),
        "Random Forest 60": RandomForestClassifier(n_estimators=60),
        "Random Forest 50": RandomForestClassifier(n_estimators=50),
        "Naive Bayes": GaussianNB(),
        "SVM linear": SVC(kernel='linear', probability=True),
        "SVM poly": SVC(kernel='poly', probability=True),
        "SVM rbf": SVC(kernel='rbf', probability=True),
        "SVM sigmoid": SVC(kernel='sigmoid', probability=True),
        "Logistic Regression": LogisticRegression(max_iter=500),
        "CART": DecisionTreeClassifier(),
        "AdaBoost": AdaBoostClassifier(n_estimators=50),
        "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False),
        "C4.5": DecisionTreeClassifier(criterion='entropy')  # Approximation for C4.5
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        log_results_to_excel(
            file_path=file_path,
            algorithm=model,
            accuracy=acc,
            precision=precision,
            recall=recall,
            f1_score=f1,
        )


def main():
    warnings.filterwarnings('ignore')
    # methods: lr, nb, cart, c4.5, adaboost, xgboost, rf, svm, knn
    # select_top_algorithms()
    input_folder = 'preprocess/impute_missing_values/cleaned/'
    # for file in os.listdir(input_folder):
    #     if file.endswith('.xlsx'):
    #         find_best_algorithm(os.path.join(input_folder, file))


if __name__ == "__main__":
    main()
