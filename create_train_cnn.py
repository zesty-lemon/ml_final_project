import os
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report, confusion_matrix, \
    ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from sklearn.preprocessing import label_binarize

import create_and_train_model
import constants as c
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers


def split_test_train_reformat(X_features: np.ndarray,
                              y_labels: np.ndarray,
                              test_size = 0.3,
                              random_state = 43) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    Xtrain, Xtest, ytrain, ytest = train_test_split(X_features,
                                                    y_labels,
                                                    test_size=test_size,
                                                    stratify=y_labels,
                                                    random_state=random_state)

    Xtrain = Xtrain.reshape((Xtrain.shape[0], Xtrain.shape[1], 1))
    Xtest = Xtest.reshape((Xtest.shape[0], Xtest.shape[1], 1))

    return Xtrain, Xtest, ytrain, ytest


# Generate analysis and report and save to file
def generate_model_analysis_report(Xtrain: np.ndarray,
                                   Xtest: np.ndarray,
                                   ytrain: np.ndarray,
                                   ytest: np.ndarray,
                                   model: keras.Sequential,
                                   directory: str,
                                   classes: Dict[int, str],
                                   optional_annotation: str = ""):

    # ---------------------- Generate & Save ROC Chart to Directory (Multi-class) --------------------

    # Force a consistent class order
    label_ids = sorted(classes.keys())
    n_classes = len(label_ids)

    # Binarize y_test for multi-class ROC (one-vs-rest)
    y_test_bin = label_binarize(ytest, classes=label_ids)

    # Predict probabilities for each class
    y_score = model.predict(Xtest)

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

    title = "ROC Curves (CNN Model)"
    if optional_annotation:
        title = title + "\n" +optional_annotation
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()

    # Save ROC plot
    roc_plot_filepath = os.path.join(directory, "roc_curves.png")
    plt.savefig(roc_plot_filepath)
    plt.show()
    plt.close()
    print(f"Saved ROC curves to: {roc_plot_filepath}")

    # ---------------- Generate & Save Report to Directory -------------------------

    report_path = os.path.join(directory, "1D_CNN_model_report.txt")

    # Training accuracy
    y_train_pred = model.predict(Xtrain)
    y_train_pred = np.argmax(y_train_pred, axis=-1) # model returns probabilties, we only care about best predicred class
    train_acc = accuracy_score(ytrain, y_train_pred)

    # Test accuracy
    y_test_pred = model.predict(Xtest)
    y_test_pred = np.argmax(y_test_pred, axis=-1) # model returns probabilties, we only care about best predicred class
    test_acc = accuracy_score(ytest, y_test_pred)

    print(f"Train Accuracy: {train_acc} Test Accuracy: {test_acc}")

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
    display_names = [create_and_train_model.pretty_label(classes[l]) for l in label_ids]

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

        f.write(f"Optional Annotation: {optional_annotation}")
    print(f"Saved model report to: {report_path}")


def create_new_trained_models(run_with_smoothing: bool):
    # Get the parent directory to persist trained models and reports
    directory_to_save_models = (
            "trained_models/1d_cnn/sandbox/" + create_and_train_model.generate_run_dir_name()
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
                                                                                 savgol_poly=3,
                                                                                 add_stat_features = False)
        optional_annotation = "With Savitzky–Golay Smoothing"
    else:
        # Read features in from file
        X_features, y_labels = create_and_train_model.read_and_resample_features(c.DATA_SOURCE_DIRECTORY,
                                                                                 c.CLASSES,
                                                                                 target_len=50,
                                                                                 add_stat_features = False)

    Xtrain, Xtest, ytrain, ytest = split_test_train_reformat(X_features, y_labels)

    n_timesteps, n_features = Xtrain.shape[1], Xtrain.shape[2]
    n_classes = len(c.CLASSES)

    model = keras.Sequential([
        layers.Input(shape=(n_timesteps, n_features)),

        layers.Conv1D(64, 5, padding="same", activation="relu"),
        layers.Conv1D(64, 5, padding="same", activation="relu"),
        layers.MaxPooling1D(2),

        layers.Conv1D(128, 3, padding="same", activation="relu"),
        layers.GlobalAveragePooling1D(),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(n_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    history = model.fit(Xtrain, ytrain,
                        validation_data=(Xtest, ytest),
                        epochs=200,
                        batch_size=4,
                        callbacks=[
                            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
                        ]
                        )

    plt.figure(figsize=(12, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('CNN Model Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    roc_plot_filepath = os.path.join(directory_to_save_models, "training_history.png")
    plt.savefig(roc_plot_filepath)
    plt.close()

    test_loss, test_acc = model.evaluate(Xtest, ytest, verbose=1)

    generate_model_analysis_report(Xtrain,
                                   Xtest,
                                   ytrain,
                                   ytest,
                                   model,
                                   directory = directory_to_save_models,
                                   classes = c.CLASSES)

    print(f"\nFinal Test Accuracy: {test_acc:.2%}")

if __name__ == "__main__":
    create_new_trained_models(run_with_smoothing=True)

#Todo: Move generate run dir name to util class.