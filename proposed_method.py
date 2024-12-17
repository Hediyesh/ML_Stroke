import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
from scipy.special import softmax
from sklearn.model_selection import train_test_split
import warnings
from sklearn.utils import shuffle
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Main loop for processing all files
def process_files(input_folder, sample_folder, n_neighbors=9):
    total_accuracy = total_precision = total_recall = total_f1 = 0
    aggregated_conf_matrix = np.array([[0, 0], [0, 0]])  # For binary classification
    count = 0

    for file in os.listdir(input_folder):
        if file.endswith('.xlsx'):
            data = pd.read_excel(os.path.join(input_folder, file))
            train_indices = np.load(os.path.join(sample_folder, f"train_indices_{file}.npy"), allow_pickle=True)
            test_indices = np.load(os.path.join(sample_folder, f"test_indices_{file}.npy"), allow_pickle=True)
            file_class_0 = file_class_1 = 0  # Counts for this file
            print(f"Processing file: {file}")
            for i in range(10):  # Assuming 10 splits per file
                train_idx = train_indices[i]
                test_idx = test_indices[i]

                train_data = data.loc[train_idx]
                test_data = data.loc[test_idx]

                X_train = train_data.drop('stroke', axis=1)
                y_train = train_data['stroke']
                X_test = test_data.drop('stroke', axis=1)
                y_test = test_data['stroke']

                # Count the number of instances of each class
                # file_class_0 = (y_train == 0).sum() + (y_test == 0).sum()
                # file_class_1 = (y_train == 1).sum() + (y_test == 1).sum()
                print(f"Class counts for file '{file}': Class 0 train = {(y_train == 0).sum()},"
                      f" Class 1 train = {(y_train == 1).sum()}, Class 0 test = {(y_test == 0).sum()},"
                      f" Class 1 test = {(y_test == 1).sum()}")
                accuracy, precision, recall, f1, conf_matrix = abnv_method_2(
                    X_train, X_test, y_train, y_test, n_neighbors
                )

                # Accumulate metrics
                total_accuracy += accuracy
                total_precision += precision
                total_recall += recall
                total_f1 += f1
                aggregated_conf_matrix += conf_matrix  # Sum confusion matrices
                count += 1
                print(f"Confusion Matrix:\n{conf_matrix}")

        # Averages across all files and splits
        avg_accuracy = total_accuracy / count
        avg_precision = total_precision / count
        avg_recall = total_recall / count
        avg_f1 = total_f1 / count

        print(f"Average Accuracy: {avg_accuracy}")
        print(f"Average Precision: {avg_precision}")
        print(f"Average Recall: {avg_recall}")
        print(f"Average F1-Score: {avg_f1}")


def calculate_attention_weights(distances):
    # Example: Assign weights based on inverse distance
    return 1 / (distances + 1e-6)


def calculate_attention_weights(distances):
    # Example: Assign weights based on inverse distance
    return 1 / (distances + 1e-6)


def abnv_method_2_old(X_train, X_test, y_train, y_test, n_neighbors):
    # Select specific features for distance calculation
    selected_features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level']

    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    # Find k-nearest neighbors for the test set
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(X_train_selected)
    distances, indices = knn.kneighbors(X_test_selected)

    # Assign attention weights
    weights = np.array([calculate_attention_weights(dist) for dist in distances])

    # Prepare weighted train and test sets
    weighted_train_X = np.array([X_train.iloc[indices_row].values for indices_row in indices])
    weighted_train_y = np.array([y_train.iloc[indices_row].values for indices_row in indices])

    # Aggregate weighted train samples for each test sample
    weighted_train_X = weighted_train_X.reshape(-1, X_train.shape[1])
    weighted_train_y = weighted_train_y.flatten()

    # Train Random Forest
    rf_model = RandomForestClassifier(max_depth=7, n_estimators=50, random_state=42)
    rf_model.fit(weighted_train_X, weighted_train_y)

    # Train XGBoost
    xgb_model = XGBClassifier(max_depth=3, learning_rate=0.01, n_estimators=100, random_state=42,
                              use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(weighted_train_X, weighted_train_y)

    # Train Logistic Regression
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(weighted_train_X, weighted_train_y)

    # Get predictions for each model
    rf_preds = rf_model.predict(X_test)
    xgb_preds = xgb_model.predict(X_test)
    lr_preds = lr_model.predict(X_test)
    print("Random Forest predictions:", rf_preds)
    print("XGBoost predictions:", xgb_preds)
    print("Logistic Regression predictions:", lr_preds)

    # Calculate weighted class scores
    final_preds = []
    for i in range(len(X_test)):
        weight_test = weights[i]

        # Random Forest weighted contributions
        weight_class_0_rf, weight_class_1_rf = 0, 0
        if rf_preds[i] == 0:
            weight_class_0_rf = weight_test * 1
        if rf_preds[i] == 1:
            weight_class_1_rf = weight_test * 1

        # XGBoost weighted contributions
        weight_class_0_xgb, weight_class_1_xgb = 0, 0
        if xgb_preds[i] == 0:
            weight_class_0_xgb = weight_test * 1
        if xgb_preds[i] == 1:
            weight_class_1_xgb = weight_test * 1

        # Logistic Regression weighted contributions
        weight_class_0_lr, weight_class_1_lr = 0, 0
        if lr_preds[i] == 0:
            weight_class_0_lr = weight_test * 1
        if lr_preds[i] == 1:
            weight_class_1_lr = weight_test * 1

        # Aggregate scores for both classes
        class_0_score = weight_class_0_rf + weight_class_0_xgb + weight_class_0_lr
        class_1_score = weight_class_1_rf + weight_class_1_xgb + weight_class_1_lr
        # print(class_1_score)
        # print(class_0_score)
        print(f"Weight class 0 RF: {weight_class_0_rf}")
        print(f"Weight class 1 RF: {weight_class_1_rf}")
        print(f"Weight class 0 XGB: {weight_class_0_xgb}")
        print(f"Weight class 1 XGB: {weight_class_1_xgb}")
        print(f"Weight class 0 LR: {weight_class_0_lr}")
        print(f"Weight class 1 LR: {weight_class_1_lr}")
        # Final prediction based on weighted scores
        final_preds.append(0 if np.sum(class_0_score) > np.sum(class_1_score) else 1)

    # Evaluate metrics
    accuracy = accuracy_score(y_test, final_preds)
    precision = precision_score(y_test, final_preds, zero_division=0)
    recall = recall_score(y_test, final_preds, zero_division=0)
    f1 = f1_score(y_test, final_preds, zero_division=0)
    conf_matrix = confusion_matrix(y_test, final_preds)

    return accuracy, precision, recall, f1, conf_matrix


def abnv_method_2(X_train, X_test, y_train, y_test, n_neighbors):
    # Select specific features for distance calculation
    selected_features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level']
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    # Find k-nearest neighbors for the test set
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(X_train_selected)
    distances, indices = knn.kneighbors(X_test_selected)

    # Separate neighbors into class 0 and class 1
    weights_class_0 = []
    weights_class_1 = []

    for i, neighbors in enumerate(indices):
        # Get classes of the neighbors
        neighbor_classes = y_train.iloc[neighbors]
        neighbor_distances = distances[i]

        # Calculate weights for each class
        weights_0 = [1 / (dist + 1e-6) for j, dist in enumerate(neighbor_distances) if neighbor_classes.iloc[j] == 0]
        weights_1 = [1 / (dist + 1e-6) for j, dist in enumerate(neighbor_distances) if neighbor_classes.iloc[j] == 1]

        # Sum weights for each class
        weights_class_0.append(np.sum(weights_0))
        weights_class_1.append(np.sum(weights_1))

    weights_class_0 = np.array(weights_class_0)
    weights_class_1 = np.array(weights_class_1)

    # Train models on the full training data
    rf_model = RandomForestClassifier(max_depth=7, n_estimators=50, random_state=42)
    rf_model.fit(X_train, y_train)

    xgb_model = XGBClassifier(max_depth=3, learning_rate=0.01, n_estimators=100, random_state=42,
                              use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)

    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train, y_train)

    # Get predictions for each model
    rf_preds = rf_model.predict(X_test)
    xgb_preds = xgb_model.predict(X_test)
    lr_preds = lr_model.predict(X_test)

    # Calculate weighted class scores
    final_preds = []
    for i in range(len(X_test)):
        # Aggregate model contributions
        weight_class_0_rf = weights_class_0[i] if rf_preds[i] == 0 else 0
        weight_class_1_rf = weights_class_1[i] if rf_preds[i] == 1 else 0

        weight_class_0_xgb = weights_class_0[i] if xgb_preds[i] == 0 else 0
        weight_class_1_xgb = weights_class_1[i] if xgb_preds[i] == 1 else 0

        weight_class_0_lr = weights_class_0[i] if lr_preds[i] == 0 else 0
        weight_class_1_lr = weights_class_1[i] if lr_preds[i] == 1 else 0

        # Aggregate scores
        class_0_score = weight_class_0_rf + weight_class_0_lr + weight_class_0_xgb
        class_1_score = weight_class_1_rf + weight_class_1_lr + weight_class_1_xgb
        # print(f"Weight class 0 RF: {weight_class_0_rf}")
        # print(f"Weight class 1 RF: {weight_class_1_rf}")
        # print(f"Weight class 0 XGB: {weight_class_0_xgb}")
        # print(f"Weight class 1 XGB: {weight_class_1_xgb}")
        # print(f"Weight class 0 LR: {weight_class_0_lr}")
        # print(f"Weight class 1 LR: {weight_class_1_lr}")
        # Final prediction based on weighted scores
        final_preds.append(0 if class_0_score > class_1_score else 1)

    # Evaluate metrics
    accuracy = accuracy_score(y_test, final_preds)
    precision = precision_score(y_test, final_preds, zero_division=0)
    recall = recall_score(y_test, final_preds, zero_division=0)
    f1 = f1_score(y_test, final_preds, zero_division=0)
    conf_matrix = confusion_matrix(y_test, final_preds)

    return accuracy, precision, recall, f1, conf_matrix


def cluster_and_classify(X_train, y_train, X_test, y_test, n_clusters=3, distance_threshold=5.0):
    # Apply clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X_train)
    cluster_labels = kmeans.predict(X_train)
    cluster_centers = kmeans.cluster_centers_

    # Determine cluster labels based on the ratio of stroke=1 to stroke=0
    cluster_ratios = {}
    for cluster in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        cluster_strokes = y_train.iloc[cluster_indices]
        stroke_1_ratio = np.sum(cluster_strokes == 1) / len(cluster_strokes)
        cluster_ratios[cluster] = stroke_1_ratio

    # Assign cluster labels
    sorted_clusters = sorted(cluster_ratios, key=cluster_ratios.get, reverse=True)
    stroke_1_cluster = sorted_clusters[0]  # Cluster with most stroke=1
    stroke_0_cluster = sorted_clusters[1]  # Cluster with most stroke=0
    unlabeled_cluster = sorted_clusters[2]  # Cluster with no label

    # Function to calculate distance to cluster centers
    def distance_to_centers(point, centers):
        return [np.linalg.norm(point - center) for center in centers]

    # Assign test data to clusters or mark as "outside"
    final_labels = []
    outside_cluster_indices = []

    for i, test_point in enumerate(X_test.values):  # Convert DataFrame to NumPy for efficiency
        distances = distance_to_centers(test_point, cluster_centers)
        closest_cluster = np.argmin(distances)
        min_distance = distances[closest_cluster]

        if min_distance <= distance_threshold:
            if closest_cluster == stroke_0_cluster:
                final_labels.append(0)
            elif closest_cluster == stroke_1_cluster:
                final_labels.append(1)
            else:
                outside_cluster_indices.append(i)
        else:
            outside_cluster_indices.append(i)

    # For points outside labeled clusters, use ML models
    if outside_cluster_indices:
        X_outside = X_test.iloc[outside_cluster_indices]

        # Train models on the full training data
        rf_model = RandomForestClassifier(max_depth=7, n_estimators=50, random_state=42)
        rf_model.fit(X_train, y_train)

        lr_model = LogisticRegression(random_state=42)
        lr_model.fit(X_train, y_train)

        xgb_model = XGBClassifier(max_depth=3, learning_rate=0.01, n_estimators=100, random_state=42,
                                  use_label_encoder=False, eval_metric='logloss')
        xgb_model.fit(X_train, y_train)

        # Predict with each model
        rf_preds = rf_model.predict(X_outside)
        lr_preds = lr_model.predict(X_outside)
        xgb_preds = xgb_model.predict(X_outside)

        # Majority voting
        for i, idx in enumerate(outside_cluster_indices):
            votes = [rf_preds[i], lr_preds[i], xgb_preds[i]]
            final_labels.append(max(set(votes), key=votes.count))

    # Evaluate performance
    accuracy = accuracy_score(y_test, final_labels)
    precision = precision_score(y_test, final_labels, zero_division=0)
    recall = recall_score(y_test, final_labels, zero_division=0)
    f1 = f1_score(y_test, final_labels, zero_division=0)
    conf_matrix = confusion_matrix(y_test, final_labels)

    return accuracy, precision, recall, f1, conf_matrix, stroke_1_cluster, stroke_0_cluster, unlabeled_cluster


def process_files_cluster(input_folder, sample_folder, file, n_clusters=3, distance_threshold=5.0):
    # Load data
    data = pd.read_excel(os.path.join(input_folder, file))
    train_indices = np.load(os.path.join(sample_folder, f"train_indices_{file}.npy"), allow_pickle=True)
    test_indices = np.load(os.path.join(sample_folder, f"test_indices_{file}.npy"), allow_pickle=True)

    for i in range(10):  # Assuming 10 splits per file
        train_idx = train_indices[i]
        test_idx = test_indices[i]

        train_data = data.loc[train_idx]
        test_data = data.loc[test_idx]

        X_train = train_data.drop('stroke', axis=1)
        y_train = train_data['stroke']
        X_test = test_data.drop('stroke', axis=1)
        y_test = test_data['stroke']
        # elbow_k, silhouette_k = find_best_k(X_train, max_k=10)
        # print(f"Best k (Elbow Method): {elbow_k}")
        # print(f"Best k (Silhouette Score): {silhouette_k}")
        # print(f"Processing split {i + 1} for file {file}...")

        # accuracy, precision, recall, f1, conf_matrix, stroke_1_cluster, stroke_0_cluster, unlabeled_cluster = cluster_and_classify(
        #     X_train, y_train, X_test, y_test, n_clusters=n_clusters, distance_threshold=distance_threshold
        # )
        # Call the function
        accuracy, precision, recall, f1, conf_matrix = cluster2_and_classify(
            X_train, y_train, X_test, y_test, threshold=0.9
        )
        print(f"Split {i + 1}:")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-Score: {f1}")
        print(f"Confusion Matrix:\n{conf_matrix}")
        # print(f"Stroke=1 Cluster: {stroke_1_cluster}")
        # print(f"Stroke=0 Cluster: {stroke_0_cluster}")
        # print(f"Unlabeled Cluster: {unlabeled_cluster}")


def find_best_k(X_train, max_k=10):
    """
    Determine the best number of clusters (k) using the Elbow Method and Silhouette Score.

    Parameters:
        X_train (DataFrame or ndarray): Training data for clustering.
        max_k (int): Maximum number of clusters to evaluate.

    Returns:
        best_k_elbow (int): Optimal k based on the Elbow Method.
        best_k_silhouette (int): Optimal k based on the Silhouette Score.
    """
    distortions = []
    silhouette_scores = []
    k_values = range(2, max_k + 1)  # Start with 2 clusters

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_train)

        # Calculate distortion (sum of squared distances to nearest cluster center)
        distortions.append(kmeans.inertia_)

        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_train, kmeans.labels_)
        silhouette_scores.append(silhouette_avg)

    # Elbow Method: Look for the "elbow" in the distortions plot
    distortions_diff = np.diff(distortions)
    elbow_point = np.argmax(-distortions_diff) + 2  # Offset by 2 because k_values starts at 2

    # Silhouette Score: Select k with the highest silhouette score
    best_k_silhouette = k_values[np.argmax(silhouette_scores)]

    # Plot Elbow Method
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(k_values, distortions, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Distortion')
    plt.title('Elbow Method')
    plt.axvline(elbow_point, color='red', linestyle='--', label=f'Best k (Elbow): {elbow_point}')
    plt.legend()

    # Plot Silhouette Scores
    plt.subplot(1, 2, 2)
    plt.plot(k_values, silhouette_scores, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method')
    plt.axvline(best_k_silhouette, color='green', linestyle='--', label=f'Best k (Silhouette): {best_k_silhouette}')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return elbow_point, best_k_silhouette


def cluster2_and_classify(X_train, y_train, X_test, y_test, threshold):
    """
    Cluster training data, assign labels to clusters, and classify test data based on cluster distances and voting.

    Parameters:
        X_train (DataFrame): Training features.
        y_train (Series): Training labels.
        X_test (DataFrame): Test features.
        y_test (Series): Test labels.
        threshold (float): Distance threshold to decide whether to assign a cluster label.

    Returns:
        accuracy, precision, recall, f1, conf_matrix: Performance metrics.
    """
    # Step 1: Clustering with KMeans (2 clusters)
    # kmeans = KMeans(n_clusters=2, random_state=42)
    # kmeans.fit(X_train)
    gmm = GaussianMixture(n_components=2, random_state=42)
    cluster_labels = gmm.fit_predict(X_train)  # Predict cluster assignments for training data
    cluster_centers = gmm.means_
    # cluster_centers = kmeans.cluster_centers_
    # cluster_labels = kmeans.labels_

    # Step 2: Assign cluster labels based on stroke distribution
    cluster_0_indices = np.where(cluster_labels == 0)[0]
    cluster_1_indices = np.where(cluster_labels == 1)[0]
    cluster_0_stroke_ratio = y_train.iloc[cluster_0_indices].mean()
    cluster_1_stroke_ratio = y_train.iloc[cluster_1_indices].mean()
    print(f'cluster_0_stroke_ratio:{cluster_0_stroke_ratio}')
    print(f'cluster_1_stroke_ratio:{cluster_1_stroke_ratio}')
    # Cluster with higher stroke ratio is labeled as 1
    cluster_label_map = {0: 0, 1: 1} if cluster_0_stroke_ratio < cluster_1_stroke_ratio else {0: 1, 1: 0}

    # Step 3: Calculate distances from test data to cluster centers
    distances = np.linalg.norm(X_test.values[:, None] - cluster_centers, axis=2)
    nearest_clusters = np.argmin(distances, axis=1)
    min_distances = np.min(distances, axis=1)
    print(f'min_distances:{min_distances}')
    # Assign cluster labels to test data based on distance threshold
    cluster_predictions = np.where(
        min_distances < threshold,
        [cluster_label_map[cluster] for cluster in nearest_clusters],
        -1  # Use -1 for points that are not assigned a cluster label
    )

    # Step 4: For unassigned test data (-1), use voting with RF, XGB, and LR
    unassigned_indices = np.where(cluster_predictions == -1)[0]
    assigned_indices = np.where(cluster_predictions != -1)[0]

    if len(unassigned_indices) > 0:
        rf_model = RandomForestClassifier(max_depth=7, n_estimators=50, random_state=42)
        xgb_model = XGBClassifier(max_depth=3, learning_rate=0.01, n_estimators=100, random_state=42,
                                  use_label_encoder=False, eval_metric='logloss')
        lr_model = LogisticRegression(random_state=42)

        # Train models on the training data
        rf_model.fit(X_train, y_train)
        xgb_model.fit(X_train, y_train)
        lr_model.fit(X_train, y_train)

        # Get predictions for the unassigned test data
        rf_preds = rf_model.predict(X_test.iloc[unassigned_indices])
        xgb_preds = xgb_model.predict(X_test.iloc[unassigned_indices])
        lr_preds = lr_model.predict(X_test.iloc[unassigned_indices])

        # Majority voting
        voting_preds = []
        for i in range(len(unassigned_indices)):
            votes = [rf_preds[i], xgb_preds[i], lr_preds[i]]
            voting_preds.append(1 if votes.count(1) > votes.count(0) else 0)

        # Update cluster_predictions for unassigned data
        cluster_predictions[unassigned_indices] = voting_preds

    # Step 5: Evaluate performance
    accuracy = accuracy_score(y_test, cluster_predictions)
    precision = precision_score(y_test, cluster_predictions, zero_division=0)
    recall = recall_score(y_test, cluster_predictions, zero_division=0)
    f1 = f1_score(y_test, cluster_predictions, zero_division=0)
    conf_matrix = confusion_matrix(y_test, cluster_predictions)

    return accuracy, precision, recall, f1, conf_matrix



def find_knn_and_cluster_old(X_train, y_train, X_test, y_test, feature_columns, threshold, k=10):
    X_train_selected = X_train[feature_columns]
    X_test_selected = X_test[feature_columns]
    print(f"\n--- KMeans with k={k} ---")
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_train_selected)
    # Assign clusters and find cluster distributions
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    # Map clusters to stroke=1 or stroke=0 based on majority class proportions
    cluster_distributions = {}
    cluster_percentages = {}
    for cluster in range(k):
        indices = np.where(cluster_labels == cluster)[0]
        y_cluster = y_train.iloc[indices]
        stroke_0_count = (y_cluster == 0).sum()
        stroke_1_count = (y_cluster == 1).sum()
        total_count = len(y_cluster)
        # Store distributions and percentages
        cluster_distributions[cluster] = {
            "stroke_0": stroke_0_count,
            "stroke_1": stroke_1_count,
        }
        cluster_percentages[cluster] = {
            "stroke_0": stroke_0_count / total_count if total_count > 0 else 0,
            "stroke_1": stroke_1_count / total_count if total_count > 0 else 0,
        }
    print("Cluster distributions (train data):")
    for cluster, counts in cluster_distributions.items():
        print(f"Cluster {cluster}: {counts}, Percentages: {cluster_percentages[cluster]}")
    # Identify clusters with highest proportion of stroke=1 and stroke=0
    stroke_1_cluster = max(cluster_percentages, key=lambda x: cluster_percentages[x]["stroke_1"])
    stroke_0_cluster = max(cluster_percentages, key=lambda x: cluster_percentages[x]["stroke_0"])
    print(f"Cluster with highest stroke=1 proportion: {stroke_1_cluster}")
    print(f"Cluster with highest stroke=0 proportion: {stroke_0_cluster}")
    # Assign labels to test data based on proximity to cluster centers
    test_labels = []
    for test_point in X_test_selected.values:
        distances = np.linalg.norm(cluster_centers - test_point, axis=1)
        closest_cluster = np.argmin(distances)
        # print(f'closest_cluster:{closest_cluster}')
        min_distance = distances[closest_cluster]
        # print(min_distance)
        # Check threshold before assigning label
        if min_distance < threshold:  # Assign label only if distance is below the threshold
            if closest_cluster == stroke_1_cluster:
                test_labels.append(1)
            elif closest_cluster == stroke_0_cluster:
                test_labels.append(0)
            else:
                test_labels.append(-1)
        else:
            test_labels.append(-1)  # Assign -1 if distance exceeds the threshold
    # Handle unclassified points by using other models (e.g., RF, XGB, LR) if needed
    classified_indices = [i for i, label in enumerate(test_labels) if label != -1]
    y_test_classified = y_test.iloc[classified_indices]
    test_labels_classified = np.array([test_labels[i] for i in classified_indices])
    # Calculate metrics only for the classified points
    accuracy = accuracy_score(y_test_classified, test_labels_classified)
    precision = precision_score(y_test_classified, test_labels_classified, zero_division=0)
    recall = recall_score(y_test_classified, test_labels_classified, zero_division=0)
    f1 = f1_score(y_test_classified, test_labels_classified, zero_division=0)
    conf_matrix = confusion_matrix(y_test_classified, test_labels_classified)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(test_labels)
    return accuracy, precision, recall, f1

def find_knn_and_cluster(X_train, y_train, X_test, y_test, feature_columns, n_neighbors, threshold,
                         k_range=range(10, 11)):
    # Select specific features for KNN
    X_train_selected = X_train[feature_columns]
    X_test_selected = X_test[feature_columns]

    # Find k-nearest neighbors for each test point
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(X_train_selected)
    knn_distances, knn_indices = knn.kneighbors(X_test_selected)

    # Initialize results
    overall_test_labels = []
    for test_idx, test_point in enumerate(X_test_selected.values):
        # print(f"\n--- Processing Test Point {test_idx + 1} ---")

        # Get the indices of the k-nearest neighbors
        neighbor_indices = knn_indices[test_idx]
        X_neighbors = X_train_selected.iloc[neighbor_indices]
        y_neighbors = y_train.iloc[neighbor_indices]

        # Perform KMeans clustering on the nearest neighbors
        for k in k_range:
            # print(f"  KMeans with k={k}")
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_neighbors)

            # Assign clusters and find cluster distributions
            cluster_labels = kmeans.labels_
            cluster_centers = kmeans.cluster_centers_

            # Map clusters to stroke=1 or stroke=0 based on majority class proportions
            cluster_distributions = {}
            cluster_percentages = {}
            for cluster in range(k):
                cluster_indices = np.where(cluster_labels == cluster)[0]
                y_cluster = y_neighbors.iloc[cluster_indices]
                stroke_0_count = (y_cluster == 0).sum()
                stroke_1_count = (y_cluster == 1).sum()
                total_count = len(y_cluster)

                cluster_distributions[cluster] = {
                    "stroke_0": stroke_0_count,
                    "stroke_1": stroke_1_count,
                }
                cluster_percentages[cluster] = {
                    "stroke_0": stroke_0_count / total_count if total_count > 0 else 0,
                    "stroke_1": stroke_1_count / total_count if total_count > 0 else 0,
                }

            # Identify clusters with the highest proportion of stroke=1 and stroke=0
            stroke_1_cluster = max(cluster_percentages, key=lambda x: cluster_percentages[x]["stroke_1"])
            stroke_0_cluster = max(cluster_percentages, key=lambda x: cluster_percentages[x]["stroke_0"])
            # print(f"Cluster with highest stroke=1 proportion: {stroke_1_cluster}")
            # print(f"Cluster with highest stroke=0 proportion: {stroke_0_cluster}")

            # Determine the cluster for the test point
            distances = np.linalg.norm(cluster_centers - test_point, axis=1)
            closest_cluster = np.argmin(distances)
            min_distance = distances[closest_cluster]

            # Classify the test point based on threshold and cluster association
            if min_distance < threshold:
                if closest_cluster == stroke_1_cluster:
                    overall_test_labels.append(1)
                elif closest_cluster == stroke_0_cluster:
                    overall_test_labels.append(0)
                else:
                    overall_test_labels.append(-1)
            else:
                overall_test_labels.append(-1)  # Assign -1 if the distance exceeds the threshold

    # Handle unclassified points
    classified_indices = [i for i, label in enumerate(overall_test_labels) if label != -1]
    y_test_classified = y_test.iloc[classified_indices]
    test_labels_classified = np.array([overall_test_labels[i] for i in classified_indices])

    # Calculate metrics only for classified points
    accuracy = accuracy_score(y_test_classified, test_labels_classified)
    precision = precision_score(y_test_classified, test_labels_classified, zero_division=0)
    recall = recall_score(y_test_classified, test_labels_classified, zero_division=0)
    f1 = f1_score(y_test_classified, test_labels_classified, zero_division=0)
    conf_matrix = confusion_matrix(y_test_classified, test_labels_classified)

    print("\nFinal Metrics:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    return accuracy, precision, recall, f1


def combined_method(X_train, y_train, X_test, y_test, feature_columns, n_nodes, k):
    """
    A single function that implements the entire method:
    - Splitting nodes by stroke class and selecting closest ones
    - Running classifiers (RF, LR, C4.5, XGB)
    - Running KMeans clustering
    - Combining predictions via voting
    """
    # Step 1: Split and select nodes from both classes
    stroke_1_data = X_train[y_train == 1]
    stroke_0_data = X_train[y_train == 0]

    # KMeans to find cluster centers
    kmeans_1 = KMeans(n_clusters=1, random_state=42).fit(stroke_1_data[feature_columns])
    kmeans_0 = KMeans(n_clusters=1, random_state=42).fit(stroke_0_data[feature_columns])

    # Compute distances and select closest n_nodes
    stroke_1_distances = np.linalg.norm(stroke_1_data[feature_columns] - kmeans_1.cluster_centers_, axis=1)
    stroke_0_distances = np.linalg.norm(stroke_0_data[feature_columns] - kmeans_0.cluster_centers_, axis=1)

    selected_stroke_1 = stroke_1_data.iloc[np.argsort(stroke_1_distances)[:n_nodes]]
    selected_stroke_0 = stroke_0_data.iloc[np.argsort(stroke_0_distances)[:n_nodes]]

    # Merge selected nodes
    merged_data = pd.concat([selected_stroke_1, selected_stroke_0])
    merged_labels = np.array([1] * n_nodes + [0] * n_nodes)

    # Step 2: Train classifiers
    rf = RandomForestClassifier(random_state=42).fit(merged_data[feature_columns], merged_labels)
    lr = LogisticRegression(random_state=42).fit(merged_data[feature_columns], merged_labels)
    c4_5 = DecisionTreeClassifier(random_state=42).fit(merged_data[feature_columns], merged_labels)
    xgb = XGBClassifier(random_state=42).fit(merged_data[feature_columns], merged_labels)

    # Exclude merged_data from the original training data
    rest_of_data = X_train.drop(merged_data.index)

    # Ensure the rest of the data contains the relevant features
    rest_of_data = rest_of_data[feature_columns]

    # Use the remaining data for KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42).fit(rest_of_data)

    # Assign clusters and find cluster distributions
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    # Analyze clusters (for example, map clusters to stroke=1 or stroke=0)
    cluster_distributions = {}
    cluster_percentages = {}

    for cluster in range(k):
        # Get the indices of rows corresponding to the cluster
        indices = np.where(cluster_labels == cluster)[0]
        # Use .loc with the original indices in rest_of_data to select corresponding y_train labels
        rest_of_data_indices = rest_of_data.index[indices]  # Get the actual indices in the original DataFrame
        y_cluster = y_train.loc[rest_of_data_indices]  # Select corresponding labels using .loc
        stroke_0_count = (y_cluster == 0).sum()
        stroke_1_count = (y_cluster == 1).sum()
        total_count = len(y_cluster)

        # Store distributions and percentages
        cluster_distributions[cluster] = {
            "stroke_0": stroke_0_count,
            "stroke_1": stroke_1_count,
        }
        cluster_percentages[cluster] = {
            "stroke_0": stroke_0_count / total_count if total_count > 0 else 0,
            "stroke_1": stroke_1_count / total_count if total_count > 0 else 0,
        }

    # Identify clusters with the highest proportion of stroke=1 and stroke=0
    stroke_1_cluster = max(cluster_percentages, key=lambda x: cluster_percentages[x]["stroke_1"])
    stroke_0_cluster = max(cluster_percentages, key=lambda x: cluster_percentages[x]["stroke_0"])

    # print(f"Cluster with highest stroke=1 proportion: {stroke_1_cluster}")
    # print(f"Cluster with highest stroke=0 proportion: {stroke_0_cluster}")
    # Step 4: Test predictions
    final_predictions = []
    for test_point in X_test.values:
        # Select features for distance-based clustering
        test_point_selected = test_point[np.isin(X_test.columns, feature_columns)].reshape(1, -1)

        # Get individual predictions
        rf_pred = rf.predict(test_point_selected)[0]
        lr_pred = lr.predict(test_point_selected)[0]
        c4_5_pred = c4_5.predict(test_point_selected)[0]
        xgb_pred = xgb.predict(test_point_selected)[0]

        # Calculate cluster distances and assign label
        distances = np.linalg.norm(kmeans.cluster_centers_ - test_point_selected, axis=1)
        cluster_res = stroke_1_cluster if np.argmin(distances) == stroke_1_cluster else stroke_0_cluster

        # Majority voting
        predictions = [rf_pred, c4_5_pred, xgb_pred, lr_pred, cluster_res]
        final_predictions.append(np.argmax(np.bincount(predictions)))

    # Step 5: Calculate metrics
    acc = accuracy_score(y_test, final_predictions)
    pr = precision_score(y_test, final_predictions)
    rec = recall_score(y_test, final_predictions)
    f1 = f1_score(y_test, final_predictions)
    conf_matrix = confusion_matrix(y_test, final_predictions)

    print(f"Accuracy: {acc}")
    print(f"Precision: {pr}")
    print(f"Recall: {rec}")
    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    return acc, pr, rec, f1


# this was bad
def combined_method_bad(X_train, y_train, X_test, y_test, feature_columns, n_nodes=10, k=2):
    """
    A single function that implements the entire method:
    - Splitting nodes by stroke class and selecting closest ones
    - Running KMeans clustering on the merged data
    - Running classifiers (RF, LR, C4.5, XGB) on the rest of the data
    - Combining predictions via voting
    """
    # Step 1: Split and select nodes from both classes
    stroke_1_data = X_train[y_train == 1]
    stroke_0_data = X_train[y_train == 0]

    # KMeans to find cluster centers
    kmeans_1 = KMeans(n_clusters=1, random_state=42).fit(stroke_1_data[feature_columns])
    kmeans_0 = KMeans(n_clusters=1, random_state=42).fit(stroke_0_data[feature_columns])

    # Compute distances and select closest n_nodes
    stroke_1_distances = np.linalg.norm(stroke_1_data[feature_columns] - kmeans_1.cluster_centers_, axis=1)
    stroke_0_distances = np.linalg.norm(stroke_0_data[feature_columns] - kmeans_0.cluster_centers_, axis=1)

    selected_stroke_1 = stroke_1_data.iloc[np.argsort(stroke_1_distances)[:n_nodes]]
    selected_stroke_0 = stroke_0_data.iloc[np.argsort(stroke_0_distances)[:n_nodes]]

    # Merge selected nodes
    merged_data = pd.concat([selected_stroke_1, selected_stroke_0])
    merged_labels = np.array([1] * n_nodes + [0] * n_nodes)

    # Step 2: Perform KMeans on the merged data
    kmeans = KMeans(n_clusters=k, random_state=42).fit(merged_data[feature_columns])
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    # Analyze clusters (map clusters to stroke=1 or stroke=0)
    cluster_distributions = {}
    cluster_percentages = {}

    for cluster in range(k):
        indices = np.where(cluster_labels == cluster)[0]
        y_cluster = merged_labels[indices]
        stroke_0_count = (y_cluster == 0).sum()
        stroke_1_count = (y_cluster == 1).sum()
        total_count = len(y_cluster)

        cluster_distributions[cluster] = {
            "stroke_0": stroke_0_count,
            "stroke_1": stroke_1_count,
        }
        cluster_percentages[cluster] = {
            "stroke_0": stroke_0_count / total_count if total_count > 0 else 0,
            "stroke_1": stroke_1_count / total_count if total_count > 0 else 0,
        }

    stroke_1_cluster = max(cluster_percentages, key=lambda x: cluster_percentages[x]["stroke_1"])
    stroke_0_cluster = max(cluster_percentages, key=lambda x: cluster_percentages[x]["stroke_0"])

    # Step 3: Train classifiers on the rest of the data
    rest_of_data = X_train.drop(merged_data.index)
    rest_labels = y_train.drop(merged_data.index)

    rf = RandomForestClassifier(random_state=42).fit(rest_of_data[feature_columns], rest_labels)
    lr = LogisticRegression(random_state=42).fit(rest_of_data[feature_columns], rest_labels)
    c4_5 = DecisionTreeClassifier(random_state=42).fit(rest_of_data[feature_columns], rest_labels)
    xgb = XGBClassifier(random_state=42).fit(rest_of_data[feature_columns], rest_labels)

    # Step 4: Test predictions
    final_predictions = []
    for test_point in X_test.values:
        test_point_selected = test_point[np.isin(X_test.columns, feature_columns)].reshape(1, -1)

        # Get individual predictions
        rf_pred = rf.predict(test_point_selected)[0]
        lr_pred = lr.predict(test_point_selected)[0]
        c4_5_pred = c4_5.predict(test_point_selected)[0]
        xgb_pred = xgb.predict(test_point_selected)[0]

        # Calculate cluster distances and assign label
        distances = np.linalg.norm(kmeans.cluster_centers_ - test_point_selected, axis=1)
        cluster_res = stroke_1_cluster if np.argmin(distances) == stroke_1_cluster else stroke_0_cluster

        # Majority voting
        predictions = [rf_pred, c4_5_pred, xgb_pred, lr_pred, cluster_res]
        final_predictions.append(np.argmax(np.bincount(predictions)))

    # Step 5: Calculate metrics
    acc = accuracy_score(y_test, final_predictions)
    pr = precision_score(y_test, final_predictions)
    rec = recall_score(y_test, final_predictions)
    f1 = f1_score(y_test, final_predictions)
    conf_matrix = confusion_matrix(y_test, final_predictions)

    print(f"Accuracy: {acc}")
    print(f"Precision: {pr}")
    print(f"Recall: {rec}")
    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    return acc, pr, rec, f1

def train_rf(part_train, part_labels, test_set):
    rf = RandomForestClassifier(random_state=42)
    rf.fit(part_train, part_labels)
    return rf.predict(test_set)

def train_lr(part_train, part_labels, test_set):
    lr = LogisticRegression(random_state=42)
    lr.fit(part_train, part_labels)
    return lr.predict(test_set)

def train_xgb(part_train, part_labels, test_set):
    xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb.fit(part_train, part_labels)
    return xgb.predict(test_set)

def split_and_voting(X_train, y_train, X_test, y_test):
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
    vote1_rf = train_rf(part1, part1_labels, X_test)
    vote1_xgb = train_xgb(part2, part2_labels, X_test)
    vote1_lr = train_lr(part3, part3_labels, X_test)
    vote1 = [max(vote) for vote in zip(vote1_rf, vote1_xgb, vote1_lr)]
    acc1 = accuracy_score(y_test, vote1)
    pr1 = precision_score(y_test, vote1)
    rec1 = recall_score(y_test, vote1)
    f11 = f1_score(y_test, vote1)
    conf_matrix1 = confusion_matrix(y_test, vote1)

    # print(f"Accuracy1: {acc1}")
    # print(f"Precision1: {pr1}")
    # print(f"Recall1: {rec1}")
    # print(f"F1 Score1: {f11}")
    # print(f"Confusion Matrix1:\n{conf_matrix1}")
    # Second set of votes
    vote2_rf = train_rf(part2, part2_labels, X_test)
    vote2_lr = train_lr(part1, part1_labels, X_test)
    vote2_xgb = train_xgb(part3, part3_labels, X_test)
    vote2 = [max(vote) for vote in zip(vote2_rf, vote2_lr, vote2_xgb)]
    acc2 = accuracy_score(y_test, vote2)
    pr2 = precision_score(y_test, vote2)
    rec2 = recall_score(y_test, vote2)
    f12 = f1_score(y_test, vote2)
    conf_matrix2 = confusion_matrix(y_test, vote2)

    # print(f"Accuracy2: {acc2}")
    # print(f"Precision2: {pr2}")
    # print(f"Recall2: {rec2}")
    # print(f"F1 Score2: {f12}")
    # print(f"Confusion Matrix2:\n{conf_matrix2}")
    # Third set of votes
    vote3_rf = train_rf(part3, part3_labels, X_test)
    vote3_lr = train_lr(part2, part2_labels, X_test)
    vote3_xgb = train_xgb(part1, part1_labels, X_test)
    vote3 = [max(vote) for vote in zip(vote3_rf, vote3_lr, vote3_xgb)]
    acc3 = accuracy_score(y_test, vote3)
    pr3 = precision_score(y_test, vote3)
    rec3 = recall_score(y_test, vote3)
    f13 = f1_score(y_test, vote3)
    conf_matrix3 = confusion_matrix(y_test, vote3)

    # print(f"Accuracy3: {acc3}")
    # print(f"Precision3: {pr3}")
    # print(f"Recall3: {rec3}")
    # print(f"F1 Score3: {f13}")
    # print(f"Confusion Matrix3:\n{conf_matrix3}")
    # Final voting across the three rounds
    final_votes = [max(vote) for vote in zip(vote1, vote2, vote3)]

    # Step 3: Calculate metrics
    acc = accuracy_score(y_test, final_votes)
    pr = precision_score(y_test, final_votes)
    rec = recall_score(y_test, final_votes)
    f1 = f1_score(y_test, final_votes)
    conf_matrix = confusion_matrix(y_test, final_votes)

    print(f"Accuracy: {acc}")
    print(f"Precision: {pr}")
    print(f"Recall: {rec}")
    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    return acc, pr, rec, f1


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

# Voting Function
def split_and_voting_ada_svm_lr(X_train, y_train, X_test, y_test):
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
    vote1 = [max(vote) for vote in zip(vote1_adaboost, vote1_svm, vote1_lr)]

    # Second set of votes
    vote2_adaboost = train_adaboost(part2, part2_labels, X_test)
    vote2_lr = train_lr(part1, part1_labels, X_test)
    vote2_svm = train_svm(part3, part3_labels, X_test)
    vote2 = [max(vote) for vote in zip(vote2_adaboost, vote2_lr, vote2_svm)]

    # Third set of votes
    vote3_adaboost = train_adaboost(part3, part3_labels, X_test)
    vote3_lr = train_lr(part2, part2_labels, X_test)
    vote3_svm = train_svm(part1, part1_labels, X_test)
    vote3 = [max(vote) for vote in zip(vote3_adaboost, vote3_lr, vote3_svm)]

    # Final voting across the three rounds
    final_votes = [max(vote) for vote in zip(vote1, vote2, vote3)]

    # Step 3: Calculate metrics
    acc = accuracy_score(y_test, final_votes)
    pr = precision_score(y_test, final_votes)
    rec = recall_score(y_test, final_votes)
    f1 = f1_score(y_test, final_votes)
    conf_matrix = confusion_matrix(y_test, final_votes)

    print(f"Accuracy: {acc}")
    print(f"Precision: {pr}")
    print(f"Recall: {rec}")
    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    return acc, pr, rec, f1

def main():
    warnings.filterwarnings('ignore')
    # Usage example
    # folder = 'split_test_train/smote_results/'
    # folder = 'split_test_train/smote_balanced_results/'
    # process_files_for_smote(folder)

    # Replace with your actual folder paths
    input_folder = 'preprocess/impute_missing_values/cleaned/'
    sample_folder = 'split_test_train/sample_results/'
    # process_files(input_folder, sample_folder)
    # process_files(input_folder, sample_folder, 29)
    # file = 'bmi_glu_median_bmi_median_smoke_k4.xlsx'
    file = 'bmi_glu_mean_bmi_knn_smoke_k4.xlsx'
    # process_files_cluster(input_folder, sample_folder, file)
    data = pd.read_excel(os.path.join(input_folder, file))
    train_indices = np.load(os.path.join(sample_folder, f"train_indices_{file}.npy"), allow_pickle=True)
    test_indices = np.load(os.path.join(sample_folder, f"test_indices_{file}.npy"), allow_pickle=True)
    avg_acc = 0
    avg_rec = 0
    avg_pr = 0
    avg_f1 = 0
    for i in range(10):  # Assuming 10 splits per file
        train_idx = train_indices[i]
        test_idx = test_indices[i]

        train_data = data.loc[train_idx]
        test_data = data.loc[test_idx]

        X_train = train_data.drop('stroke', axis=1)
        y_train = train_data['stroke']
        X_test = test_data.drop('stroke', axis=1)
        y_test = test_data['stroke']

        feature_columns = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level']
        threshold = 0.3
        n_nodes = 15
        k = 20
        # acc, pr, rec, f1 = combined_method(X_train, y_train, X_test, y_test, feature_columns, n_nodes, k)find_knn_and_cluster(X_train, y_train, X_test, y_test, feature_columns, n_neighbors, threshold)
        acc, pr, rec, f1 = split_and_voting_ada_svm_lr(X_train, y_train, X_test, y_test)
        avg_acc += acc
        avg_pr += pr
        avg_rec += rec
        avg_f1 += f1
    print(f'avg_acc:{avg_acc/10}')
    print(f'avg_pr:{avg_pr/10}')
    print(f'avg_rec:{avg_rec/10}')
    print(f'avg_f1:{avg_f1/10}')

if __name__ == "__main__":
    main()
