import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict, GridSearchCV, cross_val_score  # Add cross_val_score import
from sklearn.metrics import roc_curve, roc_auc_score

# Load the USPS and gina datasets
usps_41158 = datasets.fetch_openml(data_id=41158)
usps_41964 = datasets.fetch_openml(data_id=41964)

datasets_list = [(usps_41158, "gina 41158"), (usps_41964, "USPS 41964")]

for dataset, dataset_name in datasets_list:
    print(f"\nDataset: {dataset_name}")

    # Load dataset
    X, y = dataset.data, dataset.target

    # Define decision tree classifiers for entropy and Gini index criteria
    clf_entropy = DecisionTreeClassifier(criterion='entropy')
    clf_gini = DecisionTreeClassifier(criterion='gini')

    # Define parameter grids for GridSearchCV
    parameters = [{"min_samples_leaf": [3, 6, 9, 12, 15, 18]}]

    # Create GridSearchCV objects for entropy and Gini index criteria
    tuned_dtc_entropy = GridSearchCV(clf_entropy, parameters, scoring="roc_auc", cv=10)
    tuned_dtc_gini = GridSearchCV(clf_gini, parameters, scoring="roc_auc", cv=10)

    # Fit GridSearchCV to the data for entropy criterion
    tuned_dtc_entropy.fit(X, y)

    # Fit GridSearchCV to the data for Gini index criterion
    tuned_dtc_gini.fit(X, y)

    # Get the best parameters
    best_parameter_entropy = tuned_dtc_entropy.best_params_
    best_parameter_gini = tuned_dtc_gini.best_params_
    print("\nBest Parameter for Entropy:", best_parameter_entropy)
    print("Best Parameter for Gini Index:", best_parameter_gini)

    # Perform cross-validation and obtain prediction probabilities with the best parameters
    y_scores_entropy = cross_val_predict(tuned_dtc_entropy, X, y, method="predict_proba", cv=10)
    y_scores_gini = cross_val_predict(tuned_dtc_gini, X, y, method="predict_proba", cv=10)

    # Convert target variable to binary integers (0 and 1)
    y_true_binary = (y == '1').astype(int)

    # Calculate AUC (Area Under Curve) for entropy criterion
    auc_score_entropy = roc_auc_score(y_true_binary, y_scores_entropy[:, 1])
    print("AUC Score for Entropy:", auc_score_entropy)

    # Calculate AUC (Area Under Curve) for Gini index criterion
    auc_score_gini = roc_auc_score(y_true_binary, y_scores_gini[:, 1])
    print("AUC Score for Gini Index:", auc_score_gini)

    # Perform 10-fold cross-validation and print the results
    cv_scores_entropy = cross_val_score(tuned_dtc_entropy, X, y_true_binary, cv=10, scoring='roc_auc')
    cv_scores_gini = cross_val_score(tuned_dtc_gini, X, y_true_binary, cv=10, scoring='roc_auc')
    print("Cross-validation AUC scores (Entropy):", cv_scores_entropy)
    print("Cross-validation AUC scores (Gini Index):", cv_scores_gini)

    # Plotting the ROC curves
    fpr_entropy, tpr_entropy, _ = roc_curve(y_true_binary, y_scores_entropy[:, 1], pos_label=1)
    fpr_gini, tpr_gini, _ = roc_curve(y_true_binary, y_scores_gini[:, 1], pos_label=1)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_entropy, tpr_entropy, label='Decision Tree (Entropy) ROC curve')
    plt.plot(fpr_gini, tpr_gini, label='Decision Tree (Gini Index) ROC curve')

    # Plotting the random classifier line
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')

    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.title(f'ROC Curve for {dataset_name}')
    plt.legend(loc="lower right")
    plt.text(0.5, 0.3, f'AUC (Entropy): {auc_score_entropy:.2f}\nAUC (Gini Index): {auc_score_gini:.2f}', horizontalalignment='center', verticalalignment='center')
    plt.grid(True)
    plt.show()
