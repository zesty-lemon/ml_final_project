import os, glob
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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
    X_features = []
    y_labels = []

    for label, val in classes.items():
        for file in glob.glob(os.path.join(root, label, "*.csv")):
            x_raw_data = read_series(file)
            x_resampled = interpolate_signal(x_raw_data, target_len=target_len)
            X_features.append(x_resampled)
            y_labels.append(val)

    X_features = np.array(X_features)
    y_labels = np.array(y_labels)
    return X_features, y_labels


# find and plot accuracy vs number of estimators
# for random forest classifier
def find_best_n_estimators(Xtrain:np.ndarray, Xtest: np.ndarray, ytrain: np.ndarray, ytest: np.ndarray):
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



# train and test a random forest model with a simple test/training split
def perform_test_train_randomforest(X_features: np.ndarray, y_labels: np.ndarray, num_estimators: int = 200, random_state: int = 42):
    # train test split
    Xtrain, Xtest, ytrain, ytest = train_test_split(X_features, y_labels, test_size=0.3, stratify=y_labels,
                                                    random_state=random_state)

    # train model
    clf = RandomForestClassifier(n_estimators=num_estimators, random_state=random_state)
    clf.fit(Xtrain, ytrain)

    # evaluate model & print report
    ypred = clf.predict(Xtest)
    print(classification_report(ytest, ypred, target_names=["regular_road", "pothole"]))

    find_best_n_estimators(Xtrain, Xtest, ytrain, ytest)


#---- Run Model -----
root = "data/gonzalez_2017/data/"
classes = {"potholes": 1, "regular_road": 0}

# read features in from file and resample them
X_features, y_labels = read_and_resample_features(root, classes, 50)

perform_test_train_randomforest(X_features, y_labels)
