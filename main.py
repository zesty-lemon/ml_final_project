import os, glob
from typing import Dict, Tuple
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.stats import randint, uniform
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import classification_report, roc_curve, auc, RocCurveDisplay
from scipy.interpolate import interp1d
from tqdm import tqdm


# read csv into NDArray, skipping the first row
def read_series(filepath) -> NDArray[np.float32]:
    return np.loadtxt(filepath, skiprows=1, delimiter=',').astype(np.float32)


# resample the raw data into a fixed length using interpolation
def interpolate_signal(x_raw_data, target_len=50):
    n = len(x_raw_data)
    if n == target_len:
        return x_raw_data
    old_idx = np.linspace(0, 1, n)
    new_idx = np.linspace(0, 1, target_len)
    f = interp1d(old_idx, x_raw_data, kind='linear')
    return f(new_idx)


# read features in from file
# split into features and labels
# resize the features to be consistent length
def read_and_resample_features(file_root: str, classes: Dict[str, int], target_len:int = 50) -> Tuple[np.ndarray, np.ndarray]:
    X_features = [] # joined array of all features for all desired classes together
    y_labels = [] # labels of features for all desired classes together
    # for each class, read its values into X_features and its labels into y_labels
    for label, val in classes.items():
        for file in glob.glob(os.path.join(file_root, label, "*.csv")):
            x_raw_data = read_series(file)
            x_resampled = interpolate_signal(x_raw_data, target_len=target_len)
            X_features.append(x_resampled)
            y_labels.append(val)

    X_features = np.array(X_features)
    y_labels = np.array(y_labels)
    return X_features, y_labels


# find and plot accuracy vs number of estimators
# for random forest classifier
def find_best_n_estimators_random_forest(Xtrain:np.ndarray, Xtest: np.ndarray, ytrain: np.ndarray, ytest: np.ndarray):
    n_estimator_val = []
    rf_score = []

    # pretty print loading bars with tqdm just for fun
    for i in tqdm(range(10, 301), desc="Training Random Forests", unit="model"):
        rf_classifier = RandomForestClassifier(n_estimators=i)
        rf_classifier.fit(Xtrain, ytrain)
        y_pred = rf_classifier.predict(Xtest)
        score = rf_classifier.score(Xtest, ytest)
        n_estimator_val.append(i)
        rf_score.append(score)

    plt.figure(figsize=(8, 4.5))
    plt.plot(n_estimator_val, rf_score, linewidth=1)
    plt.xlabel("Number of Estimator Values (integer)")
    plt.ylabel("Accuracy")
    plt.title("Random Forest Classifier Accuracy vs. Number of Estimators")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()


def assess_clf_performance(clf: RandomForestClassifier, Xtest: np.ndarray,
                           ytest: np.ndarray, classes: Dict[str, int]):
    ypred = clf.predict(Xtest)
    proba = clf.predict_proba(Xtest) # order is same as class_labels
    class_labels = clf.classes_ # [0 1 2 3 4]

    print("----- Simple Test/Train Split Random Forest Classification Report -----")
    label_to_name = {v: k for k, v in classes.items()}
    ordered_labels = sorted(label_to_name.keys())  # [0,1,2,3,4]
    target_names = [label_to_name[l] for l in ordered_labels]

    print(classification_report(
        ytest,
        ypred,
        labels=ordered_labels,
        target_names=target_names
    ))

    # which numeric label is "potholes"?
    pothole_label = classes["potholes"]  # should be whatever it is originally set to
    # find which column in `proba` corresponds to that label
    pothole_col_idx = np.where(class_labels == pothole_label)[0][0]
    # binary ground truth: pothole vs rest
    ytest_binary = (ytest == pothole_label).astype(int)
    # use the corresponding probability as the score
    y_score = proba[:, pothole_col_idx]
    # compute ROC and AUC
    fpr, tpr, thresholds = roc_curve(ytest_binary, y_score)
    roc_auc = auc(fpr, tpr)

    print(f"Pothole-vs-rest AUC: {roc_auc:.3f}")

    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, name="RandomForest (pothole vs rest)").plot()
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.title("ROC Curve: Potholes vs Rest")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


# train and test a random forest model with a simple test/training split
def train_randomforest(clf: RandomForestClassifier, X_features: np.ndarray, y_labels: np.ndarray,
                       classes: Dict[str, int], random_state: int = 42,
                       find_n_estimators: bool = False, test_size=0.3,
                       print_perf_metrics: bool = True) -> RandomForestClassifier:

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X_features, y_labels, test_size=test_size, stratify=y_labels,
        random_state=random_state
    )

    # train model
    clf.fit(Xtrain, ytrain)

    if print_perf_metrics:
        assess_clf_performance(clf, Xtest, ytest, classes)

    if find_n_estimators:
        find_best_n_estimators_random_forest(Xtrain, Xtest, ytrain, ytest)

    return clf

# perform k_fold validation on random forest
def perform_k_fold_randomforest(X_features: np.ndarray, y_labels: np.ndarray, n_estimators: int = 200, random_state: int = 42):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    scores = cross_val_score(rf, X_features, y_labels, cv=cv, scoring="accuracy")
    print(f"Cross-validation accuracy: {scores.mean():.3f} ± {scores.std():.3f}")


def run_random_param_search(X_train: np.ndarray, y_train: np.ndarray):
    print("----- Beginning Randomized Search CV -----")
    # define the estimator
    rf_classifier = RandomForestClassifier(random_state=42)

    # define the parameter distributions
    param_distributions = {
        'n_estimators': randint(low=50, high=1000),
        'max_depth': randint(low=10, high=100),
        'min_samples_split': randint(low=2, high=10),
        'max_features': ['sqrt', 'log2', None],
        'criterion': ['gini', 'entropy']
    }

    # create the RandomizedSearchCV object
    random_search = RandomizedSearchCV(
        estimator=rf_classifier,
        param_distributions=param_distributions,
        n_iter=200,  # number of parameter settings that are sampled
        cv=5,  # number of cross-validation folds
        scoring='accuracy',
        random_state=36,
        n_jobs=-1  # force to use available CPU cores
    )

    random_search.fit(X_train, y_train)

    # Access the best parameters and best score
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best score: {random_search.best_score_}")
    print("----- Completed Randomized Search CV -----")


#---- Run Model -----
root = "data/gonzalez_2017/data/"
classes = {"metal_bumps": 4,"asphalt_bumps": 3, "potholes": 2, "regular_road": 1, "worn_out_road": 0}
classes.keys()
# read features in from file and resample them
X_features, y_labels = read_and_resample_features(root, classes, 50)

# perform k-fold validation random forest
perform_k_fold_randomforest(X_features, y_labels)

# fit random forest model

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf = train_randomforest(clf, X_features, y_labels, classes)

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X_features, y_labels, test_size=0.3, stratify=y_labels,
    random_state=42
)

run_random_param_search(Xtrain, ytrain)

# for X_index in range(0,len(X_features)):
#     if y_labels[X_index] == 0:
#         plt.plot(range(0, 50), X_features[X_index], 'b')
#     if y_labels[X_index] == 1:
#         plt.plot(range(0, 50), X_features[X_index], 'r')
#     if y_labels[X_index] == 2:
#         plt.plot(range(0, 50), X_features[X_index], 'g')
#     if y_labels[X_index] == 3:
#         plt.plot(range(0, 50), X_features[X_index], 'm')
# plt.show()