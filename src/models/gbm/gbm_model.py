import logging
import os
import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Import configurations and data processing functions
from src.data_preprocessing.data_preprocessing import preprocess_data
from src.utils.config import DATA_PATHS, SAVE_DIR, TRAIN_TEST_SPLIT, MODEL_PARAMS

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_data():
    """Load data from the specified paths in the config file."""
    logging.info("Loading data...")
    phenotypic_data = pd.read_csv(DATA_PATHS["phenotypic"], encoding="ISO-8859-1")
    anat_qap = pd.read_csv(DATA_PATHS["anat_qap"], encoding="ISO-8859-1")
    dti_qap = pd.read_csv(DATA_PATHS["dti_qap"], encoding="ISO-8859-1")
    functional_qap = pd.read_csv(DATA_PATHS["functional_qap"], encoding="ISO-8859-1")
    return phenotypic_data, anat_qap, dti_qap, functional_qap

def train_gradient_boosting(X_train, X_test, y_train, y_test):
    """Train the Gradient Boosting Model with PCA and Grid Search for hyperparameter tuning."""
    logging.info("Training Gradient Boosting Model...")

    # Ensure all features are numeric by removing non-numeric columns
    X_train = X_train.select_dtypes(include=['number'])
    X_test = X_test.select_dtypes(include=['number'])

    # Handle any remaining NaN values by imputing with mean
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=50, random_state=MODEL_PARAMS['gbm']["random_state"])
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Set up pipeline with scaling and Gradient Boosting
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('gbm', GradientBoostingClassifier(random_state=MODEL_PARAMS['gbm']["random_state"]))
    ])

    # Define grid search parameters from config
    param_grid = {
        'gbm__n_estimators': MODEL_PARAMS['gbm']["n_estimators"],
        'gbm__learning_rate': MODEL_PARAMS['gbm']["learning_rate"],
        'gbm__max_depth': MODEL_PARAMS['gbm']["max_depth"],
        'gbm__min_samples_split': MODEL_PARAMS['gbm']["min_samples_split"]
    }

    # Run grid search
    grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_pca, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_
    logging.info(f"Best parameters: {grid_search.best_params_}")
    logging.info(f"Best cross-validation accuracy: {grid_search.best_score_}")

    # Evaluate on test data
    y_pred = best_model.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Test Accuracy: {accuracy}")
    logging.info("Classification Report:\n" + classification_report(y_test, y_pred))
    logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # Save model and PCA transformer
    joblib.dump(best_model, os.path.join(SAVE_DIR, "gbm_model.pkl"))
    joblib.dump(pca, os.path.join(SAVE_DIR, "gbm_pca.pkl"))
    logging.info("Model and PCA transformer saved.")

    # SHAP explanations
    explainer = shap.Explainer(best_model.named_steps['gbm'], X_train_pca)
    shap_values = explainer(X_test_pca)
    plt.figure()
    shap.summary_plot(shap_values, X_test_pca, show=False)
    plt.savefig(os.path.join(SAVE_DIR, "shap_summary.png"))
    plt.show()

if __name__ == "__main__":
    # Load and preprocess data
    phenotypic_data, anat_qap, dti_qap, functional_qap = load_data()
    X, y = preprocess_data(phenotypic_data, anat_qap, dti_qap, functional_qap)

    # Check dataset size
    sample_size = X.shape[0]
    logging.info(f"Sample size: {sample_size}")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TRAIN_TEST_SPLIT["test_size"], random_state=TRAIN_TEST_SPLIT["random_state"]
    )

    # Train the model
    train_gradient_boosting(X_train, X_test, y_train, y_test)