import logging
import os
import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from src.data_preprocessing.data_preprocessing import preprocess_data

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_data():
    """Load and preprocess data."""
    logging.info("Starting data preprocessing...")

    # Correct path to the root directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

    # Define the correct data paths
    phenotypic_data_path = os.path.join(root_dir, "data/ABIDEII_Composite_Phenotypic.csv")
    anat_qap_path = os.path.join(root_dir, "data/ABIDEII_MRI_Quality_Metrics/anat_qap.csv")
    dti_qap_path = os.path.join(root_dir, "data/ABIDEII_MRI_Quality_Metrics/dti_qap.csv")
    functional_qap_path = os.path.join(root_dir, "data/ABIDEII_MRI_Quality_Metrics/functional_qap.csv")

    # Preprocess the data using the correct paths
    X, y = preprocess_data(
        phenotypic_path=phenotypic_data_path,
        anat_qap_path=anat_qap_path,
        dti_qap_path=dti_qap_path,
        functional_qap_path=functional_qap_path
    )

    if X is None or y is None:
        logging.error("Data preprocessing failed.")
        raise FileNotFoundError("One or more data files are missing or could not be processed.")

    logging.info("Data successfully loaded and preprocessed.")
    return X, y


def train_gradient_boosting(X_train, X_test, y_train, y_test, cv_folds=5):
    """Train the Gradient Boosting Model."""
    logging.info(f"Training Gradient Boosting Model with {cv_folds}-fold Cross-Validation...")
    print(f"\n=== Training Gradient Boosting Model with {cv_folds}-fold Cross-Validation ===")

    # Ensure all features are numeric
    print(f"Original Training Data Shape: {X_train.shape}")
    print(f"Original Test Data Shape: {X_test.shape}")
    X_train = X_train.select_dtypes(include=['number'])
    X_test = X_test.select_dtypes(include=['number'])
    print(f"Numeric Training Data Shape: {X_train.shape}")
    print(f"Numeric Test Data Shape: {X_test.shape}")

    # Handle NaN values
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    print(f"Imputed Training Data Shape: {X_train.shape}")
    print(f"Imputed Test Data Shape: {X_test.shape}")

    # Map target labels to {0, 1} if necessary
    y_train_mapped = y_train.map({1: 0, 2: 1})
    y_test_mapped = y_test.map({1: 0, 2: 1})

    # Pipeline setup
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('gbm', GradientBoostingClassifier(random_state=42))
    ])

    # Grid Search Parameters
    param_grid = {
        'gbm__n_estimators': [50, 100, 150],
        'gbm__learning_rate': [0.01, 0.1, 0.2],
        'gbm__max_depth': [3, 5, 7],
        'gbm__min_samples_split': [2, 5, 10]
    }

    # Grid Search
    grid_search = GridSearchCV(pipe, param_grid, cv=cv_folds, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train_mapped)

    # Best Model
    best_model = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Score: {grid_search.best_score_}")
    logging.info(f"Best Parameters: {grid_search.best_params_}")
    logging.info(f"Best Cross-Validation Score: {grid_search.best_score_}")

    # Predictions
    y_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = best_model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test_mapped, y_pred)
    auc = roc_auc_score(y_test_mapped, y_proba)
    print(f"Test Accuracy: {accuracy}")
    print(f"AUC-ROC: {auc}")
    logging.info(f"Test Accuracy: {accuracy}")
    logging.info(f"AUC-ROC: {auc}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test_mapped, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"GBM (AUC = {auc:.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(f"roc_curve_{cv_folds}_folds.png")
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(y_test_mapped, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"confusion_matrix_{cv_folds}_folds.png")
    plt.show()

    # SHAP Explanation
    explainer = shap.Explainer(best_model.named_steps['gbm'], X_train)
    shap_values = explainer(X_test)

    # SHAP Summary Plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(f"shap_summary_{cv_folds}_folds.png")
    plt.close()

    # Save Model
    joblib.dump(best_model, f"gbm_model_{cv_folds}_folds.pkl")
    logging.info(f"Model saved as 'gbm_model_{cv_folds}_folds.pkl'.")


if __name__ == "__main__":
    try:
        # Load data
        logging.info("Loading and preprocessing data...")
        X, y = load_data()

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Training Data Shape: {X_train.shape}")
        print(f"Test Data Shape: {X_test.shape}")

        # Train with 5, 7, and 10 folds
        for folds in [5, 7, 10]:
            print(f"\n=== Training with {folds}-fold Cross-Validation ===")
            train_gradient_boosting(X_train, X_test, y_train, y_test, cv_folds=folds)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
