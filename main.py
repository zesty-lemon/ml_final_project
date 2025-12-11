import create_train_random_forest_logreg
import create_train_cnn

def main():
    # Create, Train, and Evaluate Logistic Regression and Random Forest (with random sweep parameter tuning)
    create_train_random_forest_logreg.create_new_trained_models(run_k_fold_validation=True,
                                                                run_new_simple_rf_classifier=True,
                                                                run_random_param_search=True,
                                                                run_logistic_regression=True,
                                                                run_with_smoothing=True)

    # Create, Train, and Evaluate a 1D Convolutional Neural Network
    create_train_cnn.create_new_trained_models(run_with_smoothing=True)
