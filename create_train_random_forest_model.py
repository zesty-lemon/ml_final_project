import os
from datetime import datetime
import constants as c
from typing import Dict, Tuple
import numpy as np
import sklearn.ensemble
from matplotlib import pyplot as plt
from scipy.stats import randint
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import (
    classification_report,
    roc_curve,
    auc,
    RocCurveDisplay,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)



# Get unique name for bert_embeddings directory
def generate_run_dir_name(model_selection: BERT_MODEL) -> str:
    now = datetime.now()

    day = now.strftime("%d")
    month = now.strftime("%m")
    hour = now.strftime("%H")
    minute = now.strftime("%M")
    second = now.strftime("%S")

    return f"{model_selection.name}_{day}_{month}_{hour}_{minute}_{second}"


# Read features in from file
# split into features and labels
def read_features(model_selection: BERT_MODEL) -> Tuple[np.ndarray, np.ndarray]:
    X_features = [] # joined array of all features for all desired classes together
    y_labels = [] # labels of features for all desired classes together
    # for each class, read its values into X_features and its labels into y_labels

    df_labels_embeddings = word_embeddings.get_embeddings_and_labels_for_model(model_selection = model_selection,
                                                                               use_cached_embeddings=True)

    embeddings = df_labels_embeddings["Embedding"].values
    X_features = np.vstack(embeddings).astype(np.float32)
    y_labels = df_labels_embeddings["GT"].astype(int).to_numpy()

    X_features = np.array(X_features)
    y_labels = np.array(y_labels)
    return X_features, y_labels


# Train and test a random forest model with a simple test/training split
def train_randomforest(clf: RandomForestClassifier,
                       X_features: np.ndarray,
                       y_labels: np.ndarray,
                       classes: Dict[int, str],
                       model_selection: BERT_MODEL,
                       random_state: int = 42,
                       test_size=0.3,
                       report_directory: str = None) -> RandomForestClassifier:

    print("---- BEGIN Training Random Forest Classifier ---")
    Xtrain, Xtest, ytrain, ytest = train_test_split(X_features,
                                                    y_labels,
                                                    test_size=test_size,
                                                    stratify=y_labels,
                                                    random_state=random_state)

    # Train the model
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
                                   classes,
                                   model_selection = model_selection)

    # Save the trained model to file
    joblib.dump(clf, f'{full_output_dir}/random_forest_model.joblib')

    print("---- END Training Random Forest Classifier ---")
    return clf


# Perform K-Fold validation and optionally save report
def perform_k_fold_randomforest(X_features: np.ndarray,
                                y_labels: np.ndarray,
                                report_directory: str,
                                model_selection: BERT_MODEL,
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
        f.write(f"Embedding Used: {model_selection.name}\n")
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


# run a random hyperparameter search for random forest
# return the model with the best accuracy AND save a report
# set use_dummy_model_configs to true when debugging, it will run very simplified random search (faster)
def run_random_param_search(X_train: np.ndarray,
                            y_train: np.ndarray,
                            directory: str,
                            use_dummy_model_configs: bool = False) -> RandomForestClassifier:
    # define the estimator
    rf_classifier = RandomForestClassifier(random_state=42,
                                           class_weight="balanced")

    # define the parameter distributions
    param_distributions = {
        "n_estimators": randint(100, 400),
        "max_depth": [None] + list(range(10, 61, 10)),  # none = unlimited
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
        "max_features": ['sqrt', 'log2', None],
        "bootstrap": [True, False],
        "criterion": ["gini", "entropy", "log_loss"],
    }

    # create the RandomizedSearchCV object
    random_search = RandomizedSearchCV(
        estimator=rf_classifier,
        param_distributions=param_distributions,
        n_iter=40,
        cv=5,
        scoring='accuracy',
        random_state=36,
        n_jobs=-1,
        verbose=2,
        return_train_score = True
    )

    # The random search takes a very singnificant amount of time
    # these values will let it run in ~1-2 mins instead of ~2 hours
    # useful for debugging
    if use_dummy_model_configs:
        rf_classifier = RandomForestClassifier(
            random_state=42,
            class_weight="balanced",
            n_estimators=10,
            max_depth=5
        )

        # Smaller search space
        param_distributions = {
            "n_estimators": randint(5, 15),
            "max_depth": [None, 5, 10],
            "min_samples_split": randint(2, 5),
            "min_samples_leaf": randint(1, 3),
            "max_features": ['sqrt'],
            "bootstrap": [True],
            "criterion": ["gini"],
        }

        random_search = RandomizedSearchCV(
            estimator=rf_classifier,
            param_distributions=param_distributions,
            n_iter=2,
            cv=2,
            scoring='accuracy',
            random_state=36,
            n_jobs=1,
            verbose=1,
            return_train_score = True)

    # Run the search
    random_search.fit(X_train, y_train)

    # Access the best parameters and best score
    best_params = random_search.best_params_
    best_score = random_search.best_score_

    # ---- Generate & Save Report to Directory ----
    os.makedirs(directory, exist_ok=True)
    report_path = os.path.join(directory, "random_forest_random_search_report.txt")
    # Get Results from Random Forest Searc
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


# Generate & Save a report about a fitted classifier
# Statics such as test/train accuracy & ROC curve
def generate_model_analysis_report(Xtrain: np.ndarray,
                                   Xtest: np.ndarray,
                                   ytrain: np.ndarray,
                                   ytest: np.ndarray,
                                   already_fitted_clf: RandomForestClassifier,
                                   directory: str,
                                   classes: Dict[int, str],
                                   model_selection: BERT_MODEL):
    # ---- Generate & Save ROC Chart to Directory ----
    # Get ROC/AUC and Plot It
    # Find the numeric label for “humour”
    positive_label = [k for k, v in classes.items() if v == "humour"][0]

    # Find which column that label corresponds to in predict_proba
    pos_idx = list(already_fitted_clf.classes_).index(positive_label)

    # Calculate ROC with correct index
    y_score = already_fitted_clf.predict_proba(Xtest)[:, pos_idx]
    fpr, tpr, _ = roc_curve(ytest, y_score, pos_label=positive_label) # force positive label manually
    roc_auc = auc(fpr, tpr)

    # Plot ROC
    plt.figure(figsize=(6, 6))
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, name="RandomForest").plot()
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.title("ROC Curve (Random Forest Model)\nBig Bang Theory Humour Classification")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    # Save ROC to file
    plot_filepath = os.path.join(directory, "roc_curve_random_search.png")
    plt.savefig(plot_filepath)
    plt.close()
    print(f"Saved ROC curve to: {plot_filepath}")

    # ---- Generate & Save Report to Directory ----

    # Make directory & path to store final report
    report_path = os.path.join(directory, "random_forest_model_report.txt")

    # Statistics on Class Distribution (Training Set)
    train_humour_class_count = np.sum(ytrain == c.GT_HUMOUR)
    train_non_humour_class_count = np.sum(ytrain == c.GT_NON_HUMOUR)
    training_total_classes = train_humour_class_count + train_non_humour_class_count
    training_humour_class_percent = round((train_humour_class_count / training_total_classes) * 100,1)
    training_non_humour_class_percent = round((train_non_humour_class_count / training_total_classes) * 100,1)

    # Statistics on Class Distribution (Test Set)
    test_humour_class_count = np.sum(ytest == c.GT_HUMOUR)
    test_non_humour_class_count = np.sum(ytest == c.GT_NON_HUMOUR)
    test_total_classes = test_humour_class_count + test_non_humour_class_count
    test_humour_class_percent = round((test_humour_class_count / test_total_classes) * 100, 1)
    test_non_humour_class_percent = round((test_non_humour_class_count / test_total_classes) * 100,1)

    # Training accuracy
    y_train_pred = already_fitted_clf.predict(Xtrain)
    train_acc = accuracy_score(ytrain, y_train_pred)

    # Test accuracy
    y_test_pred = already_fitted_clf.predict(Xtest)
    test_acc = accuracy_score(ytest, y_test_pred)

    # Classification Report infers label order, this forces the order to be correct
    labels = sorted(classes.keys())
    target_names = [classes[l] for l in labels]

    # Generate Classification Report
    report_str = classification_report(
        ytest,
        y_test_pred,
        labels=labels,
        target_names=target_names)

    # Generate Confusion Matrix
    label_order = sorted(classes.keys()) # force ascending label order
    conf_matrix = confusion_matrix(ytest, y_test_pred, labels=label_order)
    display_names = [classes[l] for l in label_order] # force class names in the same order as keys
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                                  display_labels=display_names)
    disp.plot(cmap="Blues")
    plt.tight_layout()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    matrix_filepath = os.path.join(directory, "confusion_matrix.png")
    plt.savefig(matrix_filepath)
    plt.close()
    print(f"Saved Confusion Matrix to: {plot_filepath}")

    # Build .txt file to save final report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("----- Random Forest Model Report -----\n")
        f.write("======================================\n\n")
        f.write(f"Embedding Used: {model_selection.name}\n")
        f.write("--------- Performance Metrics --------\n")
        f.write(f"Training accuracy: {train_acc:.4f}\n")
        f.write(f"Test accuracy: {test_acc:.4f}\n")
        f.write(f"Random Forest AUC (Test Set): {roc_auc:.3f}\n\n")
        f.write(f"Classification Report: \n{report_str}\n")
        f.write("----------- General Metrics ----------\n")
        f.write(f"Train Set Instances of Humor Class: {train_humour_class_count} ({training_humour_class_percent})\n")
        f.write(f"Train Set Instances of Non-Humor Class: {train_non_humour_class_count} ({training_non_humour_class_percent})\n")
        f.write(f"Test Set Instances of Humor Class: {test_humour_class_count} ({test_humour_class_percent})\n")
        f.write(f"Test Set Instances of Non-Humor Class: {test_non_humour_class_count} ({test_non_humour_class_percent})\n")

# run random search and save best result to file
def perform_random_param_search(X_features: np.ndarray,
                                y_labels: np.ndarray,
                                model_selection: BERT_MODEL,
                                directory: str,
                                classes: Dict[int, str],
                                use_dummy_model_configs: bool = False) -> RandomForestClassifier:
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
    clf = run_random_param_search(Xtrain, ytrain, output_dir, use_dummy_model_configs)

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
                                   classes,
                                   model_selection)

    print("----- END Randomized Search CV -----")
    return clf


# Create & Save various Random Forest Models
def create_new_trained_models(run_k_fold_validation: bool,
                              run_new_simple_rf_classifier: bool,
                              run_random_param_search: bool,
                              model_selection: BERT_MODEL,
                              use_dummy_parameters: bool = False):

    directory_to_save_models = (
            c.RANDOM_FOREST_TRAINED_MODEL_DIR_PREFIX + "sandbox/" + generate_run_dir_name(model_selection)
    )

    os.makedirs(directory_to_save_models, exist_ok=True)

    # Read features in from file
    X_features, y_labels = read_features(model_selection)

    # Perform k-fold validation random forest
    if run_k_fold_validation:
        perform_k_fold_randomforest(X_features,
                                    y_labels,
                                    model_selection=model_selection,
                                    report_directory=directory_to_save_models)

    # Fit Random Forest Model
    if run_new_simple_rf_classifier:
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf = train_randomforest(clf,
                                 X_features,
                                 y_labels,
                                 model_selection = model_selection,
                                 classes=c.CLASSES,
                                 report_directory=directory_to_save_models)

    # Perform Random Search
    if run_random_param_search:
        perform_random_param_search(X_features,
                                    y_labels,
                                    model_selection = model_selection,
                                    directory=directory_to_save_models,
                                    classes=c.CLASSES,
                                    use_dummy_model_configs=use_dummy_parameters)


if __name__ == "__main__":
    # # Create trained model with BERT embeddings
    create_new_trained_models(run_k_fold_validation=True,
                              run_new_simple_rf_classifier=True,
                              run_random_param_search=True,
                              model_selection=BERT_MODEL.BERT,
                              use_dummy_parameters=False)

    # Create trained model with Sentence Bert embeddings
    create_new_trained_models(run_k_fold_validation=True,
                              run_new_simple_rf_classifier=True,
                              run_random_param_search=True,
                              model_selection=BERT_MODEL.S_BERT,
                              use_dummy_parameters=False)