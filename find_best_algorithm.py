from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import warnings
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')


def find_best_algorithm(X_train, X_test, y_train, y_test):
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
    # Train and evaluate models
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Calculate TP, TN, FP, FN
        tn, fp, fn, tp = conf_matrix.ravel()

        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Confusion Matrix': conf_matrix,
            'True Positives (TP)': tp,
            'True Negatives (TN)': tn,
            'False Positives (FP)': fp,
            'False Negatives (FN)': fn
        })

    # Print results
    for result in results:
        print(f"Model: {result['Model']}")
        print(f"Accuracy: {result['Accuracy']}")
        print(f"Precision: {result['Precision']}")
        print(f"Recall: {result['Recall']}")
        print(f"F1 Score: {result['F1 Score']}")
        print(f"Confusion Matrix:\n{result['Confusion Matrix']}")
        print(f"True Positives (TP): {result['True Positives (TP)']}")
        print(f"True Negatives (TN): {result['True Negatives (TN)']}")
        print(f"False Positives (FP): {result['False Positives (FP)']}")
        print(f"False Negatives (FN): {result['False Negatives (FN)']}")
        print("-" * 30)


# Train Adaboost
def train_adaboost(part_train, part_labels, test_set):
    adaboost = AdaBoostClassifier(n_estimators=100, random_state=42)
    adaboost.fit(part_train, part_labels)
    return adaboost.predict(test_set)


# Train SVM
def train_svm(part_train, part_labels, test_set):
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(part_train, part_labels)
    return svm.predict(test_set)


def train_lr(part_train, part_labels, test_set):
    lr = LogisticRegression(random_state=42)
    lr.fit(part_train, part_labels)
    return lr.predict(test_set)


def split_and_voting_ada_svm_lr(X_train, X_test, y_train, y_test):
    """
    Splits the training data into 3 parts randomly, trains classifiers on different parts, and performs nested voting.
    """
    # Step 1: Shuffle and split data into three equal parts
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    part1 = X_train[:len(X_train)//3]
    part1_labels = y_train[:len(y_train)//3]

    part2 = X_train[len(X_train)//3:2*len(X_train)//3]
    part2_labels = y_train[len(y_train)//3:2*len(y_train)//3]

    part3 = X_train[2*len(X_train)//3:]
    part3_labels = y_train[2*len(y_train)//3:]

    # Step 2: Train and predict with specified classifiers for each split
    # First set of votes
    vote1_adaboost = train_adaboost(part1, part1_labels, X_test)
    vote1_svm = train_svm(part2, part2_labels, X_test)
    vote1_lr = train_lr(part3, part3_labels, X_test)
    # vote1 = [max(vote) for vote in zip(vote1_adaboost, vote1_svm, vote1_lr)]
    vote1 = [Counter(vote).most_common(1)[0][0] for vote in zip(vote1_adaboost, vote1_svm, vote1_lr)]

    # Second set of votes
    vote2_adaboost = train_adaboost(part2, part2_labels, X_test)
    vote2_lr = train_lr(part1, part1_labels, X_test)
    vote2_svm = train_svm(part3, part3_labels, X_test)
    # vote2 = [max(vote) for vote in zip(vote2_adaboost, vote2_lr, vote2_svm)]
    vote2 = [Counter(vote).most_common(1)[0][0] for vote in zip(vote2_adaboost, vote2_lr, vote2_svm)]

    # Third set of votes
    vote3_adaboost = train_adaboost(part3, part3_labels, X_test)
    vote3_lr = train_lr(part2, part2_labels, X_test)
    vote3_svm = train_svm(part1, part1_labels, X_test)
    # vote3 = [max(vote) for vote in zip(vote3_adaboost, vote3_lr, vote3_svm)]
    vote3 = [Counter(vote).most_common(1)[0][0] for vote in zip(vote3_adaboost, vote3_lr, vote3_svm)]

    # Final voting across the three rounds
    # final_votes = [max(vote) for vote in zip(vote1, vote2, vote3)]
    final_votes = [Counter(vote).most_common(1)[0][0] for vote in zip(vote1, vote2, vote3)]

    # Step 3: Calculate metrics
    acc = accuracy_score(y_test, final_votes)
    pr = precision_score(y_test, final_votes)
    rec = recall_score(y_test, final_votes)
    f1 = f1_score(y_test, final_votes)
    conf_matrix = confusion_matrix(y_test, final_votes)

    # print(f"Accuracy: {acc}")
    # print(f"Precision: {pr}")
    # print(f"Recall: {rec}")
    # print(f"F1 Score: {f1}")
    # print(f"Confusion Matrix:\n{conf_matrix}")

    return acc, pr, rec, f1, conf_matrix


def nb_on_1(X_train, X_test, y_train, y_test):
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    return acc, precision, recall, f1, conf_matrix


def custom_split_no_overlap_with_indices(X_train, y_train):
    """
    Splits the data into balanced training set.
    """
    # Separate data by class
    class_1 = X_train[y_train == 1]
    class_0 = X_train[y_train == 0]
    # Select stroke=0 for test set (same size as stroke=1 in test set)
    class_0_train = class_0.sample(n=len(class_1), random_state=42)
    # Combine stroke=1 and stroke=0 for train and test sets
    X_train_balanced = pd.concat([class_1, class_0_train])
    y_train_balanced = pd.concat([pd.Series([1] * len(class_1)), pd.Series([0] * len(class_0_train))])
    # Do not reset index to preserve the original file indices
    train_indices = X_train_balanced.index.tolist()
    return X_train_balanced, y_train_balanced, train_indices


def combine_confusion_matrices(lr_conf_matrix, custom_conf_matrix):
    """
    Combines the confusion matrices of Logistic Regression and the custom method.
    Returns a NumPy array for the combined confusion matrix.
    """
    # Print the confusion matrices for debugging
    print("Logistic Regression Confusion Matrix:")
    print(lr_conf_matrix)
    print("Custom Method Confusion Matrix:")
    print(custom_conf_matrix)
    # Extract elements from Logistic Regression confusion matrix
    tn_lr, fp_lr, fn_lr, tp_lr = lr_conf_matrix.ravel()

    # Extract elements from Custom method confusion matrix
    tn_custom, fp_custom, fn_custom, tp_custom = custom_conf_matrix.ravel()

    # Combine the values: TN and FP from LR, TP and FN from custom method
    tn_combined = tn_lr
    fp_combined = fp_lr
    fn_combined = fn_custom
    tp_combined = tp_custom

    # Create a new confusion matrix as a NumPy array
    combined_conf_matrix = np.array([[tn_combined, fp_combined],
                                     [fn_combined, tp_combined]])

    return combined_conf_matrix


def lr_and_my_method(X_train, X_test, y_train, y_test):
    # Step 1: Train Logistic Regression on the training data
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train, y_train)

    # Step 2: Make predictions on the original test set using Logistic Regression
    y_pred = log_reg.predict(X_test)

    # Step 3: Calculate confusion matrix for Logistic Regression
    log_reg_conf_matrix = confusion_matrix(y_test, y_pred)

    # Step 4: Combine True Positives (TP) and False Negatives (FN) for the custom method
    combined_indices = (y_test == 1) & ((y_pred == 1) | (y_pred == 0))
    X_test_combined = X_test[combined_indices]
    y_test_combined = y_test[combined_indices]

    # Step 5: Get balanced training set using the custom split method
    X_train_balanced, y_train_balanced, train_indices = custom_split_no_overlap_with_indices(X_train, y_train)

    # Step 6: Use Logistic Regression's training data and combined TP+FN as test data for the custom method
    acc, pr, rec, f1, custom_conf_matrix = split_and_voting_ada_svm_lr(X_train_balanced, X_test_combined, y_train_balanced, y_test_combined)
    # acc, pr, rec, f1, custom_conf_matrix = nb_on_1(X_train_balanced, X_test_combined, y_train_balanced, y_test_combined)

    # Step 7: Combine the confusion matrices of Logistic Regression and custom method
    combined_conf_matrix = combine_confusion_matrices(log_reg_conf_matrix, custom_conf_matrix)

    # Step 8: Calculate accuracy, precision, recall, and F1 score from the combined confusion matrix
    tn, fp, fn, tp = combined_conf_matrix.ravel()

    # Accuracy
    combined_acc = (tn + tp) / (tn + fp + fn + tp)

    # Precision
    combined_pr = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Recall
    combined_rec = tp / (tp + fn) if (tp + fn) > 0 else 0

    # F1 Score
    combined_f1 = 2 * (combined_pr * combined_rec) / (combined_pr + combined_rec) if (combined_pr + combined_rec) > 0 else 0

    # Step 9: Print the results in the desired format for the combined method
    # print("\nCombined Method Results:")
    # print(f"acc: {combined_acc}")
    # print(f"prc: {combined_pr}")
    # print(f"rec: {combined_rec}")
    # print(f"f1: {combined_f1}")
    # print(f"conf_matrix:\n{combined_conf_matrix[0][0]} {combined_conf_matrix[0][1]}")
    # print(f"{combined_conf_matrix[1][0]} {combined_conf_matrix[1][1]}")

    # Return the results
    return {
        'combined': (combined_acc, combined_pr, combined_rec, combined_f1, combined_conf_matrix)
    }


def plot_3d_stroke_data(file, save_path='3d_stroke_plot.jpg'):
    """
    Creates a 3D scatter plot of stroke data and saves it as a JPG file.

    Parameters:
        file (str): Path to the Excel file containing the data.
        save_path (str): Path to save the JPG file. Default is '3d_stroke_plot.jpg'.
    """
    # Step 1: Read the data from the Excel file
    data = pd.read_excel(file)

    # Step 2: Filter data for stroke=0 and stroke=1
    stroke_0 = data[data['stroke'] == 0]
    stroke_1 = data[data['stroke'] == 1]

    # Step 3: Select 3 numeric features for visualization
    feature1 = 'age'  # Replace with actual feature name
    feature2 = 'hypertension'  # Replace with actual feature name
    feature3 = 'heart_disease'  # Replace with actual feature name

    # Step 4: Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot stroke=0 data in blue
    ax.scatter(stroke_0[feature1], stroke_0[feature2], stroke_0[feature3], c='b', label='Stroke = 0')

    # Plot stroke=1 data in red
    ax.scatter(stroke_1[feature1], stroke_1[feature2], stroke_1[feature3], c='r', label='Stroke = 1')

    # Labels and title
    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    ax.set_zlabel(feature3)
    ax.set_title('3D Visualization of Stroke Data')

    # Add legend
    ax.legend()

    # Step 5: Save the plot as a JPG file
    plt.savefig(save_path, format='jpg')
    print(f"Plot saved as: {save_path}")

    # Step 6: Show the plot
    plt.show()


def random_split(X, y, data):
    train_indices, test_indices = train_test_split(data.index, test_size=0.3)
    X_train, X_test = X.loc[train_indices], X.loc[test_indices]
    y_train, y_test = y.loc[train_indices], y.loc[test_indices]
    return X_train, X_test, y_train, y_test


def proportion_split(X, y, data):
    # Calculate the count of stroke=0 and stroke=1
    stroke_counts = y.value_counts()
    print("Stroke counts in the full dataset:")
    print(stroke_counts)

    # Perform stratified split
    train_indices, test_indices = train_test_split(
        data.index,
        test_size=0.3,
        stratify=y  # Ensures the split maintains the ratio of classes
        # random_state=42
    )

    # Use the indices to split X and y
    X_train, X_test = X.loc[train_indices], X.loc[test_indices]
    y_train, y_test = y.loc[train_indices], y.loc[test_indices]

    # Check the class distribution in train and test
    print("\nClass distribution in Train:")
    print(y_train.value_counts())
    print("\nClass distribution in Test:")
    print(y_test.value_counts())
    return X_train, X_test, y_train, y_test


def main():
    # Load the dataset
    # file = 'preprocess/impute_missing_values/cleaned/bmi_glu_mean_bmi_knn_smoke_k4.xlsx'
    file = 'preprocess/impute_missing_values/cleaned/bmi_glu_mean_bmi_median_smoke_k3.xlsx'
    data = pd.read_excel(file)
    # Define features (X) and target (y)
    target_column = 'stroke'
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = proportion_split(X, y, data)
    # find_best_algorithm(X_train, X_test, y_train, y_test)
    # plot_3d_stroke_data(file)
    # avg_acc = 0
    # avg_rec = 0
    # avg_pr = 0
    # avg_f1 = 0
    # for i in range(0, 10):
    #     # Train-test split with non-overlapping indices
    #     X_train, X_test, y_train, y_test = random_split(X, y, data)
    #     # overlap = set(train_indices).intersection(set(test_indices))
    #     # print(f"Overlap between train and test indices: {len(overlap)}")
    #     # # Count occurrences of each class in train and test
    #     # y_train_counts = y_train.value_counts()
    #     # y_test_counts = y_test.value_counts()
    #     # print("Class distribution in Train:")
    #     # print(y_train_counts)
    #     # print("Class distribution in Test:")
    #     # print(y_test_counts)
    #     print(i)
    #     results = lr_and_my_method(X_train, X_test, y_train, y_test)
    #     combined_results = results['combined']
    #     # You can also print the results directly:
    #     print("\nCombined Method Results:")
    #     print(f"acc: {combined_results[0]}")
    #     print(f"prc: {combined_results[1]}")
    #     print(f"rec: {combined_results[2]}")
    #     print(f"f1: {combined_results[3]}")
    #     print(f"conf_matrix:\n{combined_results[4][0][0]} {combined_results[4][0][1]}")
    #     print(f"{combined_results[4][1][0]} {combined_results[4][1][1]}")
    #     avg_acc += combined_results[0]
    #     avg_pr += combined_results[1]
    #     avg_rec += combined_results[2]
    #     avg_f1 += combined_results[3]
    # print(f'avg_acc:{avg_acc / 10}')
    # print(f'avg_pr:{avg_pr / 10}')
    # print(f'avg_rec:{avg_rec / 10}')
    # print(f'avg_f1:{avg_f1 / 10}')


if __name__ == "__main__":
    main()
