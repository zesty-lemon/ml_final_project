import os, glob
import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def read_series(fp) -> NDArray[np.float32]:
    # skip the header
    return np.loadtxt(fp, skiprows=1, delimiter=',').astype(np.float32)

root = "data/gonzalez_2017/data/"
classes = {"potholes": 1, "regular_road": 0}

X_features = []
y_labels = []

for label, val in classes.items():
    for f in glob.glob(os.path.join(root, label, "*.csv")):
        x = read_series(f)
        feats = [x.mean(), x.std(), x.min(), x.max(), np.ptp(x), np.mean(np.abs(np.diff(x)))]
        X_features.append(feats)
        y_labels.append(val)

X_features, y_labels = np.array(X_features), np.array(y_labels)

# train test split
Xtr, Xte, ytr, yte = train_test_split(X_features, y_labels, test_size=0.3, stratify=y_labels, random_state=42)

# train model
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(Xtr, ytr)

# evaluate model
ypred = clf.predict(Xte)
print(classification_report(yte, ypred, target_names=["regular_road", "pothole"]))