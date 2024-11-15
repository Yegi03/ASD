
import logging
import joblib
import os
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

# Import configurations and data processing functions
from src.data_preprocessing.data_preprocessing import preprocess_data
from src.data_preprocessing.feature_engineering import create_features
from src.utils.config import MODEL_PARAMS, DATA_PATHS, SAVE_DIR

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_data():
    logging.info(f"Phenotypic path: {DATA_PATHS['phenotypic']}")
    logging.info(f"Anat QAP path: {DATA_PATHS['anat_qap']}")
    logging.info(f"DTI QAP path: {DATA_PATHS['dti_qap']}")
    logging.info(f"Functional QAP path: {DATA_PATHS['functional_qap']}")

    phenotypic_data = pd.read_csv(DATA_PATHS["phenotypic"], encoding="ISO-8859-1")
    anat_qap = pd.read_csv(DATA_PATHS["anat_qap"])
    dti_qap = pd.read_csv(DATA_PATHS["dti_qap"])
    functional_qap = pd.read_csv(DATA_PATHS["functional_qap"])
    return phenotypic_data, anat_qap, dti_qap, functional_qap


from sklearn.metrics import roc_auc_score


def train_xgboost(X, y):
    logging.info("Training XGBoost model with optimized hyperparameters...")

    # Map target values to 0 and 1 if they are in [1, 2]
    if set(y.unique()) == {1, 2}:
        y = y.map({1: 0, 2: 1})
        logging.info("Mapped target values: {1 -> 0, 2 -> 1}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load default XGBoost parameters and param grid from config
    xgb_default_params = MODEL_PARAMS["xgboost"]["default_params"]
    xgb_param_grid = MODEL_PARAMS["xgboost"]["param_grid"]

    # Define pipeline with PCA for dimensionality reduction
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=50)),  # Adjusted number of components to balance complexity
        ('xgb', XGBClassifier(**xgb_default_params))
    ])

    # Perform GridSearch with cross-validation
    grid_search = GridSearchCV(pipeline, xgb_param_grid, cv=10, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Retrieve best model and parameters
    best_model = grid_search.best_estimator_
    logging.info(f"Best parameters: {grid_search.best_params_}")
    logging.info(f"Best cross-validation accuracy: {grid_search.best_score_}")

    # Cross-validation on full dataset
    cv_scores = cross_val_score(best_model, X, y, cv=10, scoring='accuracy')
    logging.info(f"Cross-validation scores on full dataset: {cv_scores}")
    logging.info(f"Mean cross-validation accuracy: {cv_scores.mean()}")

    # Evaluate on the test set
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    logging.info(f"Accuracy on Test Set: {accuracy}")
    logging.info(f"AUC-ROC: {auc}")
    logging.info(f"F1-Score: {f1}")
    logging.info(f"Precision: {precision}")
    logging.info(f"Recall: {recall}")
    logging.info(f"Confusion Matrix:\n{conf_matrix}")
    logging.info("Classification Report:\n" + classification_report(y_test, y_pred))

    # Save the model
    model_path = os.path.join(SAVE_DIR, 'best_xgb_model.pkl')
    joblib.dump(best_model, model_path)
    logging.info(f"XGBoost model saved successfully to '{model_path}'")

    # SHAP Explainability
    logging.info("Generating SHAP explanations...")

    # Extract PCA components for SHAP feature naming
    pca_step = best_model.named_steps['pca']
    pca_feature_names = [f"PCA_{i}" for i in range(pca_step.n_components_)]

    # Prepare SHAP explainer
    xgb_step = best_model.named_steps['xgb']
    explainer = shap.Explainer(xgb_step, X_train)

    # Compute SHAP values
    shap_values = explainer(X_test)

    # Generate SHAP summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=pca_feature_names, show=False)
    # plt.title(f'SHAP Summary Plot for XGBoost\nModel Accuracy: {accuracy:.2f}, AUC-ROC: {auc:.2f}')
    plt.savefig(os.path.join(SAVE_DIR, 'xgb_shap_summary_plot.png'), dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Load and preprocess data
    phenotypic_data, anat_qap, dti_qap, functional_qap = load_data()
    X, y = preprocess_data(phenotypic_data, anat_qap, dti_qap, functional_qap)

    # Feature engineering
    features, _ = create_features(X, y, apply_pca=False)

    # Align features and target
    common_indices = features.index.intersection(y.index)
    features = features.loc[common_indices].reset_index(drop=True)
    target = y.loc[common_indices].reset_index(drop=True)

    # Train and save the XGBoost model
    train_xgboost(features, target)
