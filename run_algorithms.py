import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


kernels = ['linear', 'poly', 'rbf', 'sigmoid']


def evaluate_svm(X_train, y_train, X_test, y_test, kernel):
    # Initialize and train the SVM
    model = SVC(kernel=kernel, random_state=42)
    model.fit(X_train, y_train)
    # Make predictions
    y_pred = model.predict(X_test)
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=1)
    recall = recall_score(y_test, y_pred, zero_division=1)
    f1 = f1_score(y_test, y_pred, zero_division=1)
    return accuracy, precision, recall, f1


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


def evaluate_random_forest(X_train, y_train, X_test, y_test, n_estimators):
    # Initialize RandomForestClassifier
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

    # Train the model
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1


# Logistic Regression
def evaluate_logistic_regression(X_train, y_train, X_test, y_test):
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1


# CART (Decision Tree Classifier)
def evaluate_cart(X_train, y_train, X_test, y_test):
    cart_model = DecisionTreeClassifier(random_state=42)
    cart_model.fit(X_train, y_train)
    y_pred = cart_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1


# C4.5 (Decision Tree with entropy criterion)
def evaluate_c4_5(X_train, y_train, X_test, y_test):
    c4_5_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
    c4_5_model.fit(X_train, y_train)
    y_pred = c4_5_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1


# AdaBoost
def evaluate_adaboost(X_train, y_train, X_test, y_test, n_estimators):
    ada_model = AdaBoostClassifier(n_estimators=n_estimators, random_state=42)
    ada_model.fit(X_train, y_train)
    y_pred = ada_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1


# LightGBM
def evaluate_lightgbm(X_train, y_train, X_test, y_test, n_estimators):
    lgbm_model = LGBMClassifier(n_estimators=n_estimators, random_state=42)
    lgbm_model.fit(X_train, y_train)
    y_pred = lgbm_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1


# XGBoost
def evaluate_xgboost(X_train, y_train, X_test, y_test, n_estimators):
    xgb_model = XGBClassifier(n_estimators=n_estimators, use_label_encoder=False, eval_metric='logloss',
                              random_state=42)
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1


# Naive Bayes
def evaluate_naive_bayes(X_train, y_train, X_test, y_test):
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    y_pred = nb_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1


# KNN (k from 2 to 10)
def evaluate_knn(X_train, y_train, X_test, y_test, k):
    # Ensure X_train and X_test are in the correct format
    if not isinstance(X_train, (pd.DataFrame, np.ndarray)):
        X_train = pd.DataFrame(X_train)
    if not isinstance(X_test, (pd.DataFrame, np.ndarray)):
        X_test = pd.DataFrame(X_test)

    # Convert to numpy arrays for sklearn compatibility
    X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test

    # Initialize KNN with k neighbors
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)

    # Make predictions
    y_pred = knn_model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return accuracy, precision, recall, f1


def evaluate_method_for_1_and_2(method, n_estimators):
    folder_1 = 'split_test_train/smote_results/'
    folder_2 = 'split_test_train/smote_balanced_results/'
    for folder in [folder_1, folder_2]:
        for file in os.listdir(folder):
            if file.startswith('train') and file.endswith('.xlsx'):
                test_file = file.replace('_smote', '').replace('train', 'test')
                train_data = pd.read_excel(os.path.join(folder, file))
                test_data = pd.read_excel(os.path.join(folder, test_file))
                X_train = train_data.drop('stroke', axis=1)
                y_train = train_data['stroke']
                X_test = test_data.drop('stroke', axis=1)
                y_test = test_data['stroke']
                if method == "lr":
                    accuracy, precision, recall, f1 = evaluate_logistic_regression(X_train, y_train, X_test, y_test)
                    # Log results
                    log_results_to_excel(
                        file_path=os.path.join(folder, file),
                        algorithm=f"{method}",
                        accuracy=accuracy,
                        precision=precision,
                        recall=recall,
                        f1_score=f1
                    )
                elif method == "knn":
                    for knn_k in range(2, 11):
                        accuracy, precision, recall, f1 = evaluate_knn(X_train, y_train, X_test, y_test, knn_k)
                        # Log results
                        log_results_to_excel(
                            file_path=os.path.join(folder, file),
                            algorithm=f"{method}(k={knn_k})",
                            accuracy=accuracy,
                            precision=precision,
                            recall=recall,
                            f1_score=f1
                        )
                elif method == "nb":
                    accuracy, precision, recall, f1 = evaluate_naive_bayes(X_train, y_train, X_test, y_test)
                    # Log results
                    log_results_to_excel(
                        file_path=os.path.join(folder, file),
                        algorithm=f"{method}",
                        accuracy=accuracy,
                        precision=precision,
                        recall=recall,
                        f1_score=f1
                    )
                elif method == "xgboost":
                    accuracy, precision, recall, f1 = evaluate_xgboost(X_train, y_train, X_test, y_test, n_estimators)
                    # Log results
                    log_results_to_excel(
                        file_path=os.path.join(folder, file),
                        algorithm=f"{method}(n_estimators={n_estimators})",
                        accuracy=accuracy,
                        precision=precision,
                        recall=recall,
                        f1_score=f1
                    )
                elif method == "rf":
                    accuracy, precision, recall, f1 = evaluate_random_forest(X_train, y_train, X_test, y_test, n_estimators)
                    # Log results
                    log_results_to_excel(
                        file_path=os.path.join(folder, file),
                        algorithm=f"{method}(n_estimators={n_estimators})",
                        accuracy=accuracy,
                        precision=precision,
                        recall=recall,
                        f1_score=f1
                    )
                elif method == "lightgbm":
                    accuracy, precision, recall, f1 = evaluate_lightgbm(X_train, y_train, X_test, y_test, n_estimators)
                    # Log results
                    log_results_to_excel(
                        file_path=os.path.join(folder, file),
                        algorithm=f"{method}(n_estimators={n_estimators})",
                        accuracy=accuracy,
                        precision=precision,
                        recall=recall,
                        f1_score=f1
                    )
                elif method == "adaboost":
                    accuracy, precision, recall, f1 = evaluate_adaboost(X_train, y_train, X_test, y_test, n_estimators)
                    # Log results
                    log_results_to_excel(
                        file_path=os.path.join(folder, file),
                        algorithm=f"{method}(n_estimators={n_estimators})",
                        accuracy=accuracy,
                        precision=precision,
                        recall=recall,
                        f1_score=f1
                    )
                elif method == "cart":
                    accuracy, precision, recall, f1 = evaluate_cart(X_train, y_train, X_test, y_test)
                    # Log results
                    log_results_to_excel(
                        file_path=os.path.join(folder, file),
                        algorithm=f"{method}",
                        accuracy=accuracy,
                        precision=precision,
                        recall=recall,
                        f1_score=f1
                    )
                elif method == "c4.5":
                    accuracy, precision, recall, f1 = evaluate_c4_5(X_train, y_train, X_test, y_test)
                    # Log results
                    log_results_to_excel(
                        file_path=os.path.join(folder, file),
                        algorithm=f"{method}",
                        accuracy=accuracy,
                        precision=precision,
                        recall=recall,
                        f1_score=f1
                    )
                elif method == "svm":
                    for kernel in kernels:
                        accuracy, precision, recall, f1 = evaluate_svm(X_train, y_train, X_test, y_test, kernel)
                        # Log results
                        log_results_to_excel(
                            file_path=os.path.join(folder, file),
                            algorithm=f"{method}(kernel={kernel})",
                            accuracy=accuracy,
                            precision=precision,
                            recall=recall,
                            f1_score=f1
                        )

# For all methods except knn and svm
def evaluate_method_for_3(method, n_estimators):
    input_folder = 'preprocess/impute_missing_values/cleaned/'
    sample_folder = 'split_test_train/sample_results/'
    for file in os.listdir(input_folder):
        if file.endswith('.xlsx'):
            data = pd.read_excel(os.path.join(input_folder, file))
            train_indices = np.load(os.path.join(sample_folder, f"train_indices_{file}.npy"), allow_pickle=True)
            test_indices = np.load(os.path.join(sample_folder, f"test_indices_{file}.npy"), allow_pickle=True)
            total_accuracy = total_precision = total_recall = total_f1 = 0
            for i in range(10):
                train_idx = train_indices[i]
                test_idx = test_indices[i]
                train_data = data.loc[train_idx]
                test_data = data.loc[test_idx]
                X_train = train_data.drop('stroke', axis=1)
                y_train = train_data['stroke']
                X_test = test_data.drop('stroke', axis=1)
                y_test = test_data['stroke']
                accuracy, precision, recall, f1 = 0, 0, 0, 0
                if method == "lr":
                    accuracy, precision, recall, f1 = evaluate_logistic_regression(X_train, y_train, X_test, y_test)
                    # Log results
                    log_results_to_excel(
                        file_path=os.path.join(input_folder, file),
                        algorithm=f"{method}",
                        accuracy=accuracy,
                        precision=precision,
                        recall=recall,
                        f1_score=f1
                    )
                elif method == "nb":
                    accuracy, precision, recall, f1 = evaluate_naive_bayes(X_train, y_train, X_test, y_test)
                    # Log results
                    log_results_to_excel(
                        file_path=os.path.join(input_folder, file),
                        algorithm=f"{method}",
                        accuracy=accuracy,
                        precision=precision,
                        recall=recall,
                        f1_score=f1
                    )
                elif method == "xgboost":
                    accuracy, precision, recall, f1 = evaluate_xgboost(X_train, y_train, X_test, y_test, n_estimators)
                    # Log results
                    log_results_to_excel(
                        file_path=os.path.join(input_folder, file),
                        algorithm=f"{method}(n_estimators={n_estimators})",
                        accuracy=accuracy,
                        precision=precision,
                        recall=recall,
                        f1_score=f1
                    )
                elif method == "rf":
                    accuracy, precision, recall, f1 = evaluate_random_forest(X_train, y_train, X_test, y_test, n_estimators)
                    # Log results
                    log_results_to_excel(
                        file_path=os.path.join(input_folder, file),
                        algorithm=f"{method}(n_estimators={n_estimators})",
                        accuracy=accuracy,
                        precision=precision,
                        recall=recall,
                        f1_score=f1
                    )
                elif method == "lightgbm":
                    accuracy, precision, recall, f1 = evaluate_lightgbm(X_train, y_train, X_test, y_test, n_estimators)
                    # Log results
                    log_results_to_excel(
                        file_path=os.path.join(input_folder, file),
                        algorithm=f"{method}(n_estimators={n_estimators})",
                        accuracy=accuracy,
                        precision=precision,
                        recall=recall,
                        f1_score=f1
                    )
                elif method == "adaboost":
                    accuracy, precision, recall, f1 = evaluate_adaboost(X_train, y_train, X_test, y_test, n_estimators)
                    # Log results
                    log_results_to_excel(
                        file_path=os.path.join(input_folder, file),
                        algorithm=f"{method}(n_estimators={n_estimators})",
                        accuracy=accuracy,
                        precision=precision,
                        recall=recall,
                        f1_score=f1
                    )
                elif method == "cart":
                    accuracy, precision, recall, f1 = evaluate_cart(X_train, y_train, X_test, y_test)
                    # Log results
                    log_results_to_excel(
                        file_path=os.path.join(input_folder, file),
                        algorithm=f"{method}",
                        accuracy=accuracy,
                        precision=precision,
                        recall=recall,
                        f1_score=f1
                    )
                elif method == "c4.5":
                    accuracy, precision, recall, f1 = evaluate_c4_5(X_train, y_train, X_test, y_test)
                    # Log results
                    log_results_to_excel(
                        file_path=os.path.join(input_folder, file),
                        algorithm=f"{method}",
                        accuracy=accuracy,
                        precision=precision,
                        recall=recall,
                        f1_score=f1
                    )
                total_accuracy += accuracy
                total_precision += precision
                total_recall += recall
                total_f1 += f1
            # Calculate and log average metrics
            avg_accuracy = total_accuracy / 10
            avg_precision = total_precision / 10
            avg_recall = total_recall / 10
            avg_f1 = total_f1 / 10
            if method == "rf" or method == "xgboost" or method == "adaboost" or method == "lightgbm":
                log_results_to_excel(
                    file_path=os.path.join(input_folder, file),
                    algorithm=f"average_{method}(n_estimators={n_estimators})",
                    accuracy=avg_accuracy,
                    precision=avg_precision,
                    recall=avg_recall,
                    f1_score=avg_f1,
                )
            else:
                log_results_to_excel(
                    file_path=os.path.join(input_folder, file),
                    algorithm=f"average_{method}",
                    accuracy=avg_accuracy,
                    precision=avg_precision,
                    recall=avg_recall,
                    f1_score=avg_f1,
                )


def evaluate_svm_for_3():
    input_folder = 'preprocess/impute_missing_values/cleaned/'
    sample_folder = 'split_test_train/sample_results/'
    for file in os.listdir(input_folder):
        if file.endswith('.xlsx'):
            data = pd.read_excel(os.path.join(input_folder, file))
            train_indices = np.load(os.path.join(sample_folder, f"train_indices_{file}.npy"), allow_pickle=True)
            test_indices = np.load(os.path.join(sample_folder, f"test_indices_{file}.npy"), allow_pickle=True)

            for kernel in kernels:
                total_accuracy = total_precision = total_recall = total_f1 = 0

                for i in range(10):
                    train_idx = train_indices[i]
                    test_idx = test_indices[i]

                    train_data = data.loc[train_idx]
                    test_data = data.loc[test_idx]

                    X_train = train_data.drop('stroke', axis=1)
                    y_train = train_data['stroke']
                    X_test = test_data.drop('stroke', axis=1)
                    y_test = test_data['stroke']

                    accuracy, precision, recall, f1 = evaluate_svm(X_train, y_train, X_test, y_test, kernel)
                    total_accuracy += accuracy
                    total_precision += precision
                    total_recall += recall
                    total_f1 += f1

                    # Log individual sample results
                    log_results_to_excel(
                        file_path=os.path.join(input_folder, file),
                        algorithm=f"SVM(kernel={kernel})_sample_{i+1}",
                        accuracy=accuracy,
                        precision=precision,
                        recall=recall,
                        f1_score=f1
                    )

                # Calculate and log average metrics
                avg_accuracy = total_accuracy / 10
                avg_precision = total_precision / 10
                avg_recall = total_recall / 10
                avg_f1 = total_f1 / 10

                log_results_to_excel(
                    file_path=os.path.join(input_folder, file),
                    algorithm=f"average_SVM(kernel={kernel})",
                    accuracy=avg_accuracy,
                    precision=avg_precision,
                    recall=avg_recall,
                    f1_score=avg_f1,
                )


def evaluate_knn_for_3():
    input_folder = 'preprocess/impute_missing_values/cleaned/'
    sample_folder = 'split_test_train/sample_results/'
    for file in os.listdir(input_folder):
        if file.endswith('.xlsx'):
            data = pd.read_excel(os.path.join(input_folder, file))
            train_indices = np.load(os.path.join(sample_folder, f"train_indices_{file}.npy"), allow_pickle=True)
            test_indices = np.load(os.path.join(sample_folder, f"test_indices_{file}.npy"), allow_pickle=True)

            for k in range(2, 11):
                total_accuracy = total_precision = total_recall = total_f1 = 0

                for i in range(10):
                    train_idx = train_indices[i]
                    test_idx = test_indices[i]

                    train_data = data.loc[train_idx]
                    test_data = data.loc[test_idx]

                    X_train = train_data.drop('stroke', axis=1)
                    y_train = train_data['stroke']
                    X_test = test_data.drop('stroke', axis=1)
                    y_test = test_data['stroke']

                    accuracy, precision, recall, f1 = evaluate_knn(X_train, y_train, X_test, y_test, k)
                    total_accuracy += accuracy
                    total_precision += precision
                    total_recall += recall
                    total_f1 += f1

                    # Log individual sample results
                    log_results_to_excel(
                        file_path=os.path.join(input_folder, file),
                        algorithm=f"KNN(k={k})",
                        accuracy=accuracy,
                        precision=precision,
                        recall=recall,
                        f1_score=f1
                    )

                # Calculate and log average metrics
                avg_accuracy = total_accuracy / 10
                avg_precision = total_precision / 10
                avg_recall = total_recall / 10
                avg_f1 = total_f1 / 10

                log_results_to_excel(
                    file_path=os.path.join(input_folder, file),
                    algorithm=f"average_KNN(k={k})",
                    accuracy=avg_accuracy,
                    precision=avg_precision,
                    recall=avg_recall,
                    f1_score=avg_f1,
                )


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


def main():
    warnings.filterwarnings('ignore')
    # methods: lr, nb, cart, c4.5, adaboost, xgboost, lightgbm(gave error), rf, svm, knn
    n_estimators = 100
    # evaluate_method_for_1_and_2("svm", n_estimators)
    # print('done')
    # evaluate_method_for_1_and_2("knn", n_estimators)
    # print('done')
    # evaluate_method_for_3("xgboost", n_estimators)
    # print('1 2 done')
    # evaluate_svm_for_3()
    # print('done')
    # evaluate_knn_for_3()
    select_top_algorithms()


if __name__ == "__main__":
    main()
