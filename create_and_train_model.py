import os, glob
from datetime import datetime
import constants as c
from typing import Dict, Tuple
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.stats import randint
import joblib
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from scipy.interpolate import interp1d

# Get unique name for bert_embeddings directory
def generate_run_dir_name() -> str:
    now = datetime.now()

    day = now.strftime("%d")
    month = now.strftime("%m")
    hour = now.strftime("%H")
    minute = now.strftime("%M")
    second = now.strftime("%S")

    return f"Training_{day}_{month}_{hour}_{minute}_{second}"


# Helper method to lowercase labels and remove underscores
def pretty_label(s: str) -> str:
    return s.replace("_", " ").title()


# Read csv into NDArray, skipping the first row
def read_series(filepath) -> NDArray[np.float32]:
    return np.loadtxt(filepath, skiprows=1, delimiter=',').astype(np.float32)


# Resample the raw data into a fixed length using interpolation
def interpolate_signal(x_raw_data, target_len=50):
    n = len(x_raw_data)
    if n == target_len:
        return x_raw_data
    old_idx = np.linspace(0, 1, n)
    new_idx = np.linspace(0, 1, target_len)
    f = interp1d(old_idx, x_raw_data, kind='linear')
    return f(new_idx)


# Read features in from file
# Split into features and labels
# Resize the features to be consistent length
def read_and_resample_features(file_root: str,
                               classes: Dict[int, str],
                               target_len: int = 50) -> Tuple[np.ndarray, np.ndarray]:

    X_features = []  # list of all feature arrays
    y_labels = [] # list of corresponding labels

    for class_id, class_name in classes.items():
        class_folder = class_name.lower()
        pattern = os.path.join(file_root, class_folder, "*.csv")

        for file_path in glob.glob(pattern):
            x_raw_data = read_series(file_path)
            x_resampled = interpolate_signal(x_raw_data, target_len=target_len)

            X_features.append(x_resampled)
            y_labels.append(class_id)

    X_features = np.array(X_features)
    y_labels = np.array(y_labels)

    return X_features, y_labels


# Train and test a random forest model with a simple test/training split
def train_randomforest(clf: RandomForestClassifier,
                       X_features: np.ndarray,
                       y_labels: np.ndarray,
                       classes: Dict[int, str],
                       report_directory: str,
                       random_state: int = 42,
                       test_size=0.3) -> RandomForestClassifier:

    # Split into Test/Training
    Xtrain, Xtest, ytrain, ytest = train_test_split(X_features,
                                                    y_labels,
                                                    test_size=test_size,
                                                    stratify=y_labels,
                                                    random_state=random_state)

    # Train model
    clf.fit(Xtrain, ytrain)

    # Assess Performance and Save Report to file
    full_output_dir = os.path.join(report_directory, "manually_instantiated_model")
    os.makedirs(full_output_dir, exist_ok=True)
    generate_model_analysis_report(Xtrain,
                                   Xtest,
                                   ytrain,
                                   ytest,
                                   clf,
                                   full_output_dir,
                                   classes)

    return clf


# Perform K-Fold validation and save report
def perform_k_fold_randomforest(X_features: np.ndarray,
                                y_labels: np.ndarray,
                                report_directory: str,
                                n_estimators: int = 200,
                                random_state: int = 42):

    print(f"---- BEGIN Cross Validation (Random Forest) ----")
    # Run Cross Validation and get scores
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    scores = cross_val_score(rf, X_features, y_labels, cv=cv, scoring="accuracy")

    # Compute mean & STDEV
    mean_score = scores.mean()
    std_score = scores.std()

    # Generate & Save Report
    kfold_dir = os.path.join(report_directory, "k_fold_validation")
    os.makedirs(kfold_dir, exist_ok=True)
    report_path = os.path.join(kfold_dir, "random_forest_kfold_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Random Forest K-Fold Cross-Validation Report\n")
        f.write("===========================================\n\n")
        f.write(f"n_estimators: {n_estimators}\n")
        f.write(f"random_state: {random_state}\n")
        f.write(f"n_splits: {cv.get_n_splits()}\n\n")

        f.write("Fold accuracies:\n")
        for i, s in enumerate(scores, start=1):
            f.write(f"  Fold {i}: {s:.4f}\n")

        f.write("\n")
        f.write(f"Mean accuracy: {mean_score:.4f}\n")
        f.write(f"Std accuracy: {std_score:.4f}\n")

    print(f"Saved k-fold report to: {report_path}")
    print(f"---- END Cross Validation (Random Forest) ----")



# Run a random hyperparameter search for random forest
# return the model with the best accuracy AND save a report
# set use_dummy_model_configs to true when debugging, it will run very simplified random search (faster)
def perform_random_param_search(X_train: np.ndarray,
                                y_train: np.ndarray,
                                directory: str) -> RandomForestClassifier:
    # Define the estimator
    rf_classifier = RandomForestClassifier(random_state=42,
                                           class_weight="balanced")

    # Define the parameter distributions
    param_distributions = {
        "n_estimators": randint(100, 400),
        "max_depth": [None] + list(range(10, 61, 10)),  # none = unlimited
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
        "max_features": ['sqrt', 'log2', None],
        "bootstrap": [True, False],
        "criterion": ["gini", "entropy", "log_loss"],
    }

    # Create the RandomizedSearchCV object
    random_search = RandomizedSearchCV(
        estimator=rf_classifier,
        param_distributions=param_distributions,
        n_iter=40,
        cv=5,
        scoring='accuracy',
        random_state=36,
        n_jobs=-1,
        verbose=2, # debug flag
        return_train_score = True
    )

    # Run the search
    random_search.fit(X_train, y_train)

    # Access the best parameters and best score
    best_params = random_search.best_params_
    best_score = random_search.best_score_

    # ---- Generate & Save Report to Directory ----
    os.makedirs(directory, exist_ok=True)
    report_path = os.path.join(directory, "random_forest_random_search_report.txt")

    # Get Results from Random Forest Search
    cv_results = random_search.cv_results_

    # Scores on Test data
    mean_test_scores = cv_results["mean_test_score"]
    std_test_scores = cv_results["std_test_score"]

    # Scores on Train data
    mean_train_scores = cv_results["mean_train_score"]
    std_train_scores = cv_results["std_train_score"]

    # All Model Params
    params_list = cv_results["params"]

    # Sort configurations from best to worst
    sorted_indices = np.argsort(mean_test_scores)[::-1]

    # Generate & Save Final Report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Random Forest RandomizedSearchCV Report\n")
        f.write("======================================\n\n")

        f.write("Best configuration\n")
        f.write("------------------\n")
        for k, v in best_params.items():
            f.write(f"{k}: {v}\n")
        f.write(f"\nBest mean CV accuracy: {best_score:.4f}\n\n")

        f.write("Search space\n")
        f.write("-----------\n")
        f.write(str(param_distributions) + "\n\n")

        f.write("All tried configurations (sorted by mean accuracy)\n")
        f.write("-------------------------------------------------\n")
        for rank, idx in enumerate(sorted_indices, start=1):
            f.write(f"Rank {rank}\n")
            f.write(f"  mean_train_accuracy: {mean_train_scores[idx]:.4f}\n")
            f.write(f"  std_train_accuracy:  {std_train_scores[idx]:.4f}\n")
            f.write(f"  mean_test_accuracy:  {mean_test_scores[idx]:.4f}\n")
            f.write(f"  std_test_accuracy:   {std_test_scores[idx]:.4f}\n")
            f.write(f"  params:              {params_list[idx]}\n\n")

    print(f"Saved random search report to: {report_path}")
    return random_search.best_estimator_


# run random search and save best result to file
def orchestrate_random_param_search(X_features: np.ndarray,
                                y_labels: np.ndarray,
                                directory: str,
                                classes: Dict[int, str]) -> RandomForestClassifier:
    print("----- BEGIN Randomized Search CV -----")

    # Split into Test/Train sets
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X_features,
        y_labels,
        test_size=0.3,
        stratify=y_labels,  # classes are unbalanced this keeps proportions (prevents a split from having 0 of a class)
        random_state=42,
    )

    # Make the bert_embeddings directory to store model & report
    output_dir = directory + "/random_search_model/"
    os.makedirs(output_dir, exist_ok=True)

    # Run random search for best configuration of parameters
    clf = perform_random_param_search(Xtrain, ytrain, output_dir)

    # Save best model to file
    joblib.dump(clf, f'{output_dir}/random_forest_model.joblib')

    # Generate Report about our best model found with Random Search
    os.makedirs(output_dir, exist_ok=True)
    generate_model_analysis_report(Xtrain,
                                   Xtest,
                                   ytrain,
                                   ytest,
                                   clf,
                                   output_dir,
                                   classes)

    print("----- END Randomized Search CV -----")
    return clf


# Generate analysis and report and save to file
def generate_model_analysis_report(Xtrain: np.ndarray,
                                   Xtest: np.ndarray,
                                   ytrain: np.ndarray,
                                   ytest: np.ndarray,
                                   already_fitted_clf: RandomForestClassifier,
                                   directory: str,
                                   classes: Dict[int, str]):

    # Ensure output directory exists
    os.makedirs(directory, exist_ok=True)

    # ---------------------- Generate & Save ROC Chart to Directory (Multi-class) --------------------

    # Force a consistent class order
    label_ids = sorted(classes.keys())
    n_classes = len(label_ids)

    # Binarize y_test for multi-class ROC (one-vs-rest)
    y_test_bin = label_binarize(ytest, classes=label_ids)

    # Predict probabilities for each class
    y_score = already_fitted_clf.predict_proba(Xtest)  # shape: (n_samples, n_classes)

    # Compute ROC/AUC Statistics=
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i, class_id in enumerate(label_ids):
        fpr[class_id], tpr[class_id], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[class_id] = auc(fpr[class_id], tpr[class_id])

    # Plot all class ROC curves on the same figure
    plt.figure(figsize=(7, 7))
    for class_id in label_ids:
        class_name = classes[class_id].lower()
        plt.plot(
            fpr[class_id],
            tpr[class_id],
            label=f"{class_name} (AUC = {roc_auc[class_id]:.2f})",
        )

    # Diagonal line for random chance
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (Random Forest Model)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()

    # Save ROC plot
    roc_plot_filepath = os.path.join(directory, "roc_curves.png")
    plt.savefig(roc_plot_filepath)
    plt.close()
    print(f"Saved ROC curves to: {roc_plot_filepath}")

    # ---------------- Generate & Save Report to Directory -------------------------

    report_path = os.path.join(directory, "random_forest_model_report.txt")

    # Training accuracy
    y_train_pred = already_fitted_clf.predict(Xtrain)
    train_acc = accuracy_score(ytrain, y_train_pred)

    # Test accuracy
    y_test_pred = already_fitted_clf.predict(Xtest)
    test_acc = accuracy_score(ytest, y_test_pred)

    # Classification Report (force label order)
    target_names = [classes[l].lower() for l in label_ids]
    report_str = classification_report(
        ytest,
        y_test_pred,
        labels=label_ids,
        target_names=target_names,
    )

    # Generate Confusion Matrix
    conf_matrix = confusion_matrix(ytest, y_test_pred, labels=label_ids)

    # Make labels pretty
    display_names = [pretty_label(classes[l]) for l in label_ids]

    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix,
        display_labels=display_names,
    )

    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.tight_layout()
    plt.subplots_adjust(left=0.25, bottom=0.35)

    # Save Confusion Matrix
    matrix_filepath = os.path.join(directory, "confusion_matrix.png")
    plt.savefig(matrix_filepath)
    plt.close()
    print(f"Saved Confusion Matrix to: {matrix_filepath}")

    # Build & Save Final Report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("----- Random Forest Model Report -----\n")
        f.write("======================================\n\n")
        f.write("--------- Performance Metrics --------\n")
        f.write(f"Training accuracy: {train_acc:.4f}\n")
        f.write(f"Test accuracy: {test_acc:.4f}\n\n")

        f.write("Per-class AUC (one-vs-rest):\n")
        for class_id in label_ids:
            class_name = classes[class_id].lower()
            f.write(f"  {class_name}: {roc_auc[class_id]:.3f}\n")
        f.write("\n")

        f.write("Classification Report:\n")
        f.write(report_str)
        f.write("\n")

    print(f"Saved model report to: {report_path}")


# Create & Save various Random Forest Models
def create_new_trained_models(run_k_fold_validation: bool,
                              run_new_simple_rf_classifier: bool,
                              run_random_param_search: bool):

    # Get the parent directory to persist trained models and reports
    directory_to_save_models = (
            c.RANDOM_FOREST_TRAINED_MODEL_DIR_PREFIX + "sandbox/" + generate_run_dir_name()
    )
    os.makedirs(directory_to_save_models, exist_ok=True)

    # Read features in from file
    X_features, y_labels = read_and_resample_features(c.DATA_SOURCE_DIRECTORY, c.CLASSES, 50)

    # Perform k-fold validation random forest
    if run_k_fold_validation:
        perform_k_fold_randomforest(X_features,
                                    y_labels,
                                    report_directory=directory_to_save_models)

    # Fit Random Forest Model
    if run_new_simple_rf_classifier:
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf = train_randomforest(clf,
                                 X_features,
                                 y_labels,
                                 classes=c.CLASSES,
                                 report_directory=directory_to_save_models)

    # Perform Random Search
    if run_random_param_search:
        orchestrate_random_param_search(X_features,
                                        y_labels,
                                        directory=directory_to_save_models,
                                        classes=c.CLASSES)


if __name__ == "__main__":
    create_new_trained_models(run_k_fold_validation=True,
                              run_new_simple_rf_classifier=True,
                              run_random_param_search=True)
