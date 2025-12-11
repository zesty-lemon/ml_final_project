import joblib
import numpy as np

import constants as c
import create_train_random_forest_logreg

from sklearn.ensemble import RandomForestClassifier

# Paths to trained models (trained_models directory)
random_forest_model_path = "trained_models/random_forest/final/Training_29_11_13_51_00/random_search_model/random_forest_model.joblib"


# Unpack trained classifier from trained_models/random_forest directory
def get_trained_model(path_to_model: str) -> RandomForestClassifier:
    # model retrival inside method to force typing hints to work correctly
    trained_classifier = joblib.load(path_to_model)
    return trained_classifier


# Load classifiers into memory
rf_classifier = get_trained_model(random_forest_model_path)

# also have to append and extract features
def predict_class_of_data(input_filepath: str, enable_feature_engineering: bool = False) -> bool:
    # Step 1: Get the Classifier
    classifier = get_trained_model(random_forest_model_path)

    # Step 2: Open data to be classified
    input_features = create_train_random_forest_logreg.read_series(input_filepath)

    # Step 3: Extract & Append Statistical Features
    stat_features = create_train_random_forest_logreg.extract_stat_features(input_features)

    # Step 4: Interpolate Input to specified length
    stat_features_resampled = create_train_random_forest_logreg.interpolate_signal(stat_features, target_len=50)

    # Step 5: Optionally Append features to input vector
    if enable_feature_engineering:
        full_feature_vector = np.hstack([stat_features_resampled, stat_features])

    else:
        full_feature_vector = stat_features_resampled

    # Step 6: Reshape input
    reshaped_vector = full_feature_vector.reshape(1,-1)

    # Step 7: Classify Input
    input_classification = classifier.predict(reshaped_vector)

    predicted_class = c.CLASSES.get(input_classification[0])

    print(f"Predicted Class: {predicted_class}")

    return input_classification


if __name__ == "__main__":
    # ---------- Example Usage ----------
    # Take some input data, and classify the road surface

    # Example 1: Running with Pothole
    path_to_pothole_data = "data/gonzalez_2017/data/potholes/5e62c2dc-d40c-453e-a649-a0f42dbeeca4.csv"
    print("Should Predict Potholes:")
    predict_class_of_data(path_to_pothole_data)

    # Example 2: Running with Asphalt Bumps
    path_to_asphalt_bump_data = "data/gonzalez_2017/data/asphalt_bumps/2aeb745f-ed26-4e38-8330-bdd7db37fb3b.csv"
    print("Should Asphalt Bumps:")
    predict_class_of_data(path_to_asphalt_bump_data)