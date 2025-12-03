import os
import numpy as np
import pandas as pd
import create_and_train_model
import constants as c



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
                                                                                 target_len=50,
                                                                                 use_savgol=True,
                                                                                 savgol_window=9,
                                                                                 savgol_poly=3)
        optional_annotation = "With Savitzky–Golay Smoothing"
    else:
        # Read features in from file
        X_features, y_labels = create_and_train_model.read_and_resample_features(c.DATA_SOURCE_DIRECTORY,
                                                                                 c.CLASSES,
                                                                                 target_len=50)



if __name__ == "__main__":
    # Run with Savitzky–Golay Smoothing
    create_new_trained_models(run_with_smoothing=True)
#Todo: Move generate run dir name to util class.