
import os
import pandas as pd
import numpy as np
import joblib
import logging
import shap
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Import configurations
from src.utils.config import DATA_PATHS, SAVE_DIR, TRAIN_TEST_SPLIT, MODEL_PARAMS, TARGET_LABEL

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_data():
    logging.info("Loading data...")
    phenotypic_data = pd.read_csv(DATA_PATHS["phenotypic"], encoding="ISO-8859-1")
    anat_qap = pd.read_csv(DATA_PATHS["anat_qap"])
    dti_qap = pd.read_csv(DATA_PATHS["dti_qap"])
    functional_qap = pd.read_csv(DATA_PATHS["functional_qap"])

    for df in [phenotypic_data, anat_qap, dti_qap, functional_qap]:
        df.columns = df.columns.str.upper()

    return phenotypic_data, anat_qap, dti_qap, functional_qap

def preprocess_data(phenotypic_data, anat_qap, dti_qap, functional_qap):
    logging.info("Preprocessing data...")
    data = pd.merge(phenotypic_data, anat_qap, on="SUB_ID", how="outer", suffixes=("", "_anat"))
    data = pd.merge(data, dti_qap, on="SUB_ID", how="outer", suffixes=("", "_dti"))
    data = pd.merge(data, functional_qap, on="SUB_ID", how="outer", suffixes=("", "_func"))

    y = data[TARGET_LABEL].replace({1: 0, 2: 1})
    data = data.drop(columns=[TARGET_LABEL])

    drop_columns = ['SUB_ID', 'NDAR_GUID', 'SITE_ID', 'SITE_ID_ANAT', 'SESSION', 'SERIES', 'UNIQUE_ID',
                    'SITE_ID_DTI', 'SESSION_DTI', 'SERIES_DTI', 'UNIQUE_ID_DTI', 'SITE_ID_FUNC', 'SESSION_FUNC',
                    'SERIES_FUNC', 'UNIQUE_ID_FUNC']
    data = data.drop(columns=drop_columns, errors='ignore')
    data = data.dropna(axis=1, how='all')

    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].astype('category').cat.codes

    valid_rows = y.notna()
    X = data[valid_rows]
    y = y[valid_rows]

    return X, y

def train_svm(X, y):
    logging.info("Training SVM model with PCA and SMOTE...")

    # Impute missing values and scale features
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    pca = PCA(n_components=50)

    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)
    X_pca = pca.fit_transform(X_scaled)

    # SMOTE for balancing
    smote = SMOTE()
    X_res, y_res = smote.fit_resample(X_pca, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res,
                                                        test_size=TRAIN_TEST_SPLIT["test_size"],
                                                        random_state=TRAIN_TEST_SPLIT["random_state"])

    # SVM model with grid search and cross-validation
    param_grid = {
        'C': MODEL_PARAMS['svm']['C'],
        'gamma': MODEL_PARAMS['svm']['gamma'],
        'kernel': MODEL_PARAMS['svm']['kernel']
    }
    grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_svm = grid_search.best_estimator_
    logging.info(f"Best SVM Parameters: {grid_search.best_params_}")

    # Cross-validation on the full dataset to check for stability
    cv_scores = cross_val_score(best_svm, X_res, y_res, cv=10, scoring='accuracy')
    logging.info(f"Cross-validation scores on full dataset: {cv_scores}")
    logging.info(f"Mean cross-validation accuracy: {np.mean(cv_scores)}")

    # Evaluate the model
    y_pred = best_svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"SVM Test Accuracy: {accuracy}")
    logging.info("Classification Report:\n" + classification_report(y_test, y_pred))
    logging.info("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)))

    # Save the model and transformers
    joblib.dump(imputer, os.path.join(SAVE_DIR, "svm_imputer.pkl"))
    joblib.dump(scaler, os.path.join(SAVE_DIR, "svm_scaler.pkl"))
    joblib.dump(pca, os.path.join(SAVE_DIR, "svm_pca.pkl"))
    joblib.dump(best_svm, os.path.join(SAVE_DIR, "best_svm_model.pkl"))
    logging.info("Model and transformers saved successfully.")

    return best_svm, X_train, X_test, accuracy

def generate_shap_explanations(best_svm, X_train, X_test, accuracy):
    logging.info("Generating SHAP explanations for SVM model with limited data...")
    explainer = shap.KernelExplainer(best_svm.predict, shap.sample(X_train, 100))  # Reduced background data
    shap_values = explainer.shap_values(X_test[:50])  # Explain a subset of test samples

    # Plot SHAP summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_test[:50], feature_names=[f'PCA_{i}' for i in range(X_test.shape[1])], show=False)
    plt.title(f'SHAP Summary Plot for SVM\nModel Accuracy: {accuracy:.2f}')
    plt.savefig(os.path.join(SAVE_DIR, 'svm_shap_summary_plot.png'), dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Load and preprocess data
    phenotypic_data, anat_qap, dti_qap, functional_qap = load_data()
    X, y = preprocess_data(phenotypic_data, anat_qap, dti_qap, functional_qap)

    # Train and evaluate SVM
    best_svm, X_train, X_test, accuracy = train_svm(X, y)

    # Generate SHAP explanations
    generate_shap_explanations(best_svm, X_train, X_test, accuracy)
