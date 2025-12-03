import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import create_and_train_model
import constants as c
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def split_test_train_reformat(X_features: np.ndarray,
                              y_labels: np.ndarray,
                              test_size = 0.3,
                              random_state = 43):

    Xtrain, Xtest, ytrain, ytest = train_test_split(X_features,
                                                    y_labels,
                                                    test_size=test_size,
                                                    stratify=y_labels,
                                                    random_state=random_state)

    Xtrain = Xtrain.reshape((Xtrain.shape[0], Xtrain.shape[1], 1))
    Xtest = Xtest.reshape((Xtest.shape[0], Xtest.shape[1], 1))


def create_new_trained_models(run_with_smoothing: bool):
    # Get the parent directory to persist trained models and reports
    directory_to_save_models = (
            c.RANDOM_FOREST_TRAINED_MODEL_DIR_PREFIX + "sandbox/" + create_and_train_model.generate_run_dir_name()
    )
    os.makedirs(directory_to_save_models, exist_ok=True)

    optional_annotation = ""
    if run_with_smoothing:
        # Read in features & apply Savitzky–Golay smoothing
        X_features, y_labels = create_and_train_model.read_and_resample_features(c.DATA_SOURCE_DIRECTORY,
                                                                                 c.CLASSES,
                                                                                 target_len=100,
                                                                                 use_savgol=True,
                                                                                 savgol_window=9,
                                                                                 savgol_poly=3)
        optional_annotation = "With Savitzky–Golay Smoothing"
    else:
        # Read features in from file
        X_features, y_labels = create_and_train_model.read_and_resample_features(c.DATA_SOURCE_DIRECTORY,
                                                                                 c.CLASSES,
                                                                                 target_len=100)

    split_test_train_reformat(X_features, y_labels)


if __name__ == "__main__":
    # Run with Savitzky–Golay Smoothing
    create_new_trained_models(run_with_smoothing=True)
#Todo: Move generate run dir name to util class.