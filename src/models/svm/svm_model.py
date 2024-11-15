# # import os
# # import pandas as pd
# # import numpy as np
# # import joblib
# # import logging
# # import shap
# # import matplotlib.pyplot as plt
# # from sklearn.svm import SVC
# # from sklearn.pipeline import Pipeline
# # from sklearn.impute import SimpleImputer
# # from sklearn.decomposition import PCA
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
# # from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
# # from imblearn.over_sampling import SMOTE
# #
# # # Import configurations
# # from src.utils.config import DATA_PATHS, SAVE_DIR, TRAIN_TEST_SPLIT, MODEL_PARAMS, TARGET_LABEL
# #
# # # Set up logging
# # logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# #
# #
# # def load_data():
# #     logging.info("Loading data...")
# #     phenotypic_data = pd.read_csv(DATA_PATHS["phenotypic"], encoding="ISO-8859-1")
# #     anat_qap = pd.read_csv(DATA_PATHS["anat_qap"])
# #     dti_qap = pd.read_csv(DATA_PATHS["dti_qap"])
# #     functional_qap = pd.read_csv(DATA_PATHS["functional_qap"])
# #
# #     for df in [phenotypic_data, anat_qap, dti_qap, functional_qap]:
# #         df.columns = df.columns.str.upper()
# #
# #     return phenotypic_data, anat_qap, dti_qap, functional_qap
# #
# #
# # def preprocess_data(phenotypic_data, anat_qap, dti_qap, functional_qap):
# #     logging.info("Preprocessing data...")
# #     data = pd.merge(phenotypic_data, anat_qap, on="SUB_ID", how="outer", suffixes=("", "_anat"))
# #     data = pd.merge(data, dti_qap, on="SUB_ID", how="outer", suffixes=("", "_dti"))
# #     data = pd.merge(data, functional_qap, on="SUB_ID", how="outer", suffixes=("", "_func"))
# #
# #     y = data[TARGET_LABEL].replace({1: 0, 2: 1})
# #     data = data.drop(columns=[TARGET_LABEL])
# #
# #     drop_columns = ['SUB_ID', 'NDAR_GUID', 'SITE_ID', 'SITE_ID_ANAT', 'SESSION', 'SERIES', 'UNIQUE_ID',
# #                     'SITE_ID_DTI', 'SESSION_DTI', 'SERIES_DTI', 'UNIQUE_ID_DTI', 'SITE_ID_FUNC', 'SESSION_FUNC',
# #                     'SERIES_FUNC', 'UNIQUE_ID_FUNC']
# #     data = data.drop(columns=drop_columns, errors='ignore')
# #     data = data.dropna(axis=1, how='all')
# #
# #     for col in data.select_dtypes(include=['object']).columns:
# #         data[col] = data[col].astype('category').cat.codes
# #
# #     valid_rows = y.notna()
# #     X = data[valid_rows]
# #     y = y[valid_rows]
# #
# #     return X, y
# #
# #
# # def train_svm(X, y):
# #     logging.info("Training SVM model with PCA and SMOTE...")
# #
# #     # Impute missing values and scale features
# #     imputer = SimpleImputer(strategy="mean")
# #     scaler = StandardScaler()
# #     pca = PCA(n_components=50)
# #
# #     X_imputed = imputer.fit_transform(X)
# #     X_scaled = scaler.fit_transform(X_imputed)
# #     X_pca = pca.fit_transform(X_scaled)
# #
# #     # Preserve PCA component names
# #     pca_feature_names = [f"PCA_{i+1}" for i in range(pca.n_components_)]
# #
# #     # SMOTE for balancing
# #     smote = SMOTE()
# #     X_res, y_res = smote.fit_resample(X_pca, y)
# #
# #     # Train-test split
# #     X_train, X_test, y_train, y_test = train_test_split(
# #         X_res, y_res, test_size=TRAIN_TEST_SPLIT["test_size"], random_state=TRAIN_TEST_SPLIT["random_state"]
# #     )
# #
# #     # SVM model with grid search and cross-validation
# #     param_grid = {
# #         'C': MODEL_PARAMS['svm']['C'],
# #         'gamma': MODEL_PARAMS['svm']['gamma'],
# #         'kernel': MODEL_PARAMS['svm']['kernel']
# #     }
# #     grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
# #     grid_search.fit(X_train, y_train)
# #
# #     best_svm = grid_search.best_estimator_
# #     logging.info(f"Best SVM Parameters: {grid_search.best_params_}")
# #
# #     # Cross-validation on the full dataset to check for stability
# #     cv_scores = cross_val_score(best_svm, X_res, y_res, cv=10, scoring='accuracy')
# #     logging.info(f"Cross-validation scores on full dataset: {cv_scores}")
# #     logging.info(f"Mean cross-validation accuracy: {np.mean(cv_scores)}")
# #
# #     # Evaluate the model
# #     y_pred = best_svm.predict(X_test)
# #     y_proba = best_svm.predict_proba(X_test)[:, 1]  # Probability for the positive class
# #     accuracy = accuracy_score(y_test, y_pred)
# #     auc = roc_auc_score(y_test, y_proba)
# #
# #     logging.info(f"SVM Test Accuracy: {accuracy}")
# #     logging.info(f"AUC-ROC: {auc}")
# #     logging.info("Classification Report:\n" + classification_report(y_test, y_pred))
# #     logging.info("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)))
# #
# #     # Plot ROC Curve
# #     fpr, tpr, _ = roc_curve(y_test, y_proba)
# #     plt.figure()
# #     plt.plot(fpr, tpr, label=f"SVM (AUC = {auc:.2f})")
# #     plt.xlabel("False Positive Rate")
# #     plt.ylabel("True Positive Rate")
# #     plt.title("ROC Curve")
# #     plt.legend(loc="lower right")
# #     plt.savefig(os.path.join(SAVE_DIR, "svm_roc_curve.png"))
# #     plt.show()
# #
# #     # Save the model and transformers
# #     joblib.dump(imputer, os.path.join(SAVE_DIR, "svm_imputer.pkl"))
# #     joblib.dump(scaler, os.path.join(SAVE_DIR, "svm_scaler.pkl"))
# #     joblib.dump(pca, os.path.join(SAVE_DIR, "svm_pca.pkl"))
# #     joblib.dump(best_svm, os.path.join(SAVE_DIR, "best_svm_model.pkl"))
# #     logging.info("Model and transformers saved successfully.")
# #
# #     # Return for SHAP analysis
# #     return best_svm, X_train, X_test, y_train, y_test, pca_feature_names
# #
# #
# # def generate_shap_explanations(best_svm, X_train, X_test, y_train, y_test, pca_feature_names):
# #     logging.info("Generating SHAP explanations for SVM model...")
# #
# #     # Use a subset of the training data for SHAP background
# #     explainer = shap.KernelExplainer(best_svm.predict_proba, X_train[:100])
# #     shap_values = explainer.shap_values(X_test[:50])  # Explain a subset of test samples
# #
# #     # Plot SHAP summary plot
# #     plt.figure()
# #     shap.summary_plot(shap_values[1], X_test[:50], feature_names=pca_feature_names, show=False)  # Class 1
# #     plt.title("SHAP Summary Plot - SVM")
# #     plt.savefig(os.path.join(SAVE_DIR, "svm_shap_summary_plot.png"), dpi=300, bbox_inches='tight')
# #     plt.show()
# #
# #
# # if __name__ == "__main__":
# #     # Load and preprocess data
# #     phenotypic_data, anat_qap, dti_qap, functional_qap = load_data()
# #     X, y = preprocess_data(phenotypic_data, anat_qap, dti_qap, functional_qap)
# #
# #     # Train and evaluate SVM
# #     best_svm, X_train, X_test, y_train, y_test, pca_feature_names = train_svm(X, y)
# #
# #     # Generate SHAP explanations
# #     generate_shap_explanations(best_svm, X_train, X_test, y_train, y_test, pca_feature_names)
#
#
#
#
# import os
# import pandas as pd
# import numpy as np
# import joblib
# import logging
# import shap
# import matplotlib.pyplot as plt
# from sklearn.svm import SVC
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
# from imblearn.over_sampling import SMOTE
#
# # Import configurations
# from src.utils.config import DATA_PATHS, SAVE_DIR, TRAIN_TEST_SPLIT, MODEL_PARAMS, TARGET_LABEL
#
# # Set up logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
#
#
# def load_data():
#     logging.info("Loading data...")
#     phenotypic_data = pd.read_csv(DATA_PATHS["phenotypic"], encoding="ISO-8859-1")
#     anat_qap = pd.read_csv(DATA_PATHS["anat_qap"])
#     dti_qap = pd.read_csv(DATA_PATHS["dti_qap"])
#     functional_qap = pd.read_csv(DATA_PATHS["functional_qap"])
#
#     for df in [phenotypic_data, anat_qap, dti_qap, functional_qap]:
#         df.columns = df.columns.str.upper()
#
#     return phenotypic_data, anat_qap, dti_qap, functional_qap
#
#
# def preprocess_data(phenotypic_data, anat_qap, dti_qap, functional_qap):
#     logging.info("Preprocessing data...")
#     data = pd.merge(phenotypic_data, anat_qap, on="SUB_ID", how="outer", suffixes=("", "_anat"))
#     data = pd.merge(data, dti_qap, on="SUB_ID", how="outer", suffixes=("", "_dti"))
#     data = pd.merge(data, functional_qap, on="SUB_ID", how="outer", suffixes=("", "_func"))
#
#     y = data[TARGET_LABEL].replace({1: 0, 2: 1})
#     data = data.drop(columns=[TARGET_LABEL])
#
#     drop_columns = ['SUB_ID', 'NDAR_GUID', 'SITE_ID', 'SITE_ID_ANAT', 'SESSION', 'SERIES', 'UNIQUE_ID',
#                     'SITE_ID_DTI', 'SESSION_DTI', 'SERIES_DTI', 'UNIQUE_ID_DTI', 'SITE_ID_FUNC', 'SESSION_FUNC',
#                     'SERIES_FUNC', 'UNIQUE_ID_FUNC']
#     data = data.drop(columns=drop_columns, errors='ignore')
#     data = data.dropna(axis=1, how='all')
#
#     for col in data.select_dtypes(include=['object']).columns:
#         data[col] = data[col].astype('category').cat.codes
#
#     valid_rows = y.notna()
#     X = data[valid_rows]
#     y = y[valid_rows]
#
#     return X, y
#
#
# def train_svm(X, y):
#     logging.info("Training SVM model with PCA and SMOTE...")
#
#     # Impute missing values and scale features
#     imputer = SimpleImputer(strategy="mean")
#     scaler = StandardScaler()
#     pca = PCA(n_components=50)
#
#     X_imputed = imputer.fit_transform(X)
#     X_scaled = scaler.fit_transform(X_imputed)
#     X_pca = pca.fit_transform(X_scaled)
#
#     # Preserve PCA component names
#     pca_feature_names = [f"PCA_{i+1}" for i in range(pca.n_components_)]
#
#     # SMOTE for balancing
#     smote = SMOTE()
#     X_res, y_res = smote.fit_resample(X_pca, y)
#
#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_res, y_res, test_size=TRAIN_TEST_SPLIT["test_size"], random_state=TRAIN_TEST_SPLIT["random_state"]
#     )
#
#     # SVM model with grid search and cross-validation
#     param_grid = {
#         'C': MODEL_PARAMS['svm']['C'],
#         'gamma': MODEL_PARAMS['svm']['gamma'],
#         'kernel': MODEL_PARAMS['svm']['kernel']
#     }
#     grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
#     grid_search.fit(X_train, y_train)
#
#     best_svm = grid_search.best_estimator_
#     logging.info(f"Best SVM Parameters: {grid_search.best_params_}")
#
#     # Cross-validation on the full dataset to check for stability
#     cv_scores = cross_val_score(best_svm, X_res, y_res, cv=10, scoring='accuracy')
#     logging.info(f"Cross-validation scores on full dataset: {cv_scores}")
#     logging.info(f"Mean cross-validation accuracy: {np.mean(cv_scores)}")
#
#     # Evaluate the model
#     y_pred = best_svm.predict(X_test)
#     y_proba = best_svm.predict_proba(X_test)[:, 1]  # Probability for the positive class
#     accuracy = accuracy_score(y_test, y_pred)
#     auc = roc_auc_score(y_test, y_proba)
#
#     logging.info(f"SVM Test Accuracy: {accuracy}")
#     logging.info(f"AUC-ROC: {auc}")
#     logging.info("Classification Report:\n" + classification_report(y_test, y_pred))
#     logging.info("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)))
#
#     # Plot ROC Curve
#     fpr, tpr, _ = roc_curve(y_test, y_proba)
#     plt.figure()
#     plt.plot(fpr, tpr, label=f"SVM (AUC = {auc:.2f})")
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.title("ROC Curve")
#     plt.legend(loc="lower right")
#     plt.savefig(os.path.join(SAVE_DIR, "svm_roc_curve.png"))
#     plt.show()
#
#     # Save the model and transformers
#     joblib.dump(imputer, os.path.join(SAVE_DIR, "svm_imputer.pkl"))
#     joblib.dump(scaler, os.path.join(SAVE_DIR, "svm_scaler.pkl"))
#     joblib.dump(pca, os.path.join(SAVE_DIR, "svm_pca.pkl"))
#     joblib.dump(best_svm, os.path.join(SAVE_DIR, "best_svm_model.pkl"))
#     logging.info("Model and transformers saved successfully.")
#
#     # Return for SHAP analysis
#     return best_svm, X_train, X_test, y_train, y_test, pca_feature_names
#
# def generate_shap_explanations(best_svm, X_train_pca, X_test_pca, pca_feature_names):
#     logging.info("Generating SHAP explanations for SVM model...")
#
#     # KernelExplainer expects the model output to match the feature space
#     explainer = shap.KernelExplainer(best_svm.predict, X_train_pca[:100])  # Subset for background
#     shap_values = explainer.shap_values(X_test_pca[:50])  # Compute SHAP values for the test set
#
#     # SHAP values should now match the shape of X_test_pca
#     logging.info(f"SHAP values shape: {shap_values.shape}")
#     logging.info(f"X_test_pca shape: {X_test_pca[:50].shape}")
#
#     # Verify shapes
#     if shap_values[0].shape != X_test_pca[:50].shape:
#         raise ValueError(
#             f"Shape mismatch: SHAP values have shape {shap_values[0].shape}, "
#             f"but X_test_pca has shape {X_test_pca[:50].shape}."
#         )
#
#     # Generate SHAP summary plot
#     plt.figure()
#     shap.summary_plot(
#         shap_values[0],  # Use the SHAP values corresponding to the first class
#         X_test_pca[:50],
#         feature_names=pca_feature_names,
#         show=False
#     )
#     plt.title("SHAP Summary Plot - SVM")
#     plt.savefig(os.path.join(SAVE_DIR, "svm_shap_summary_plot.png"), dpi=300, bbox_inches='tight')
#     plt.show()
#
#
# if __name__ == "__main__":
#     # Load and preprocess data
#     phenotypic_data, anat_qap, dti_qap, functional_qap = load_data()
#     X, y = preprocess_data(phenotypic_data, anat_qap, dti_qap, functional_qap)
#
#     # Train and evaluate SVM
#     best_svm, X_train_pca, X_test_pca, y_train, y_test, pca_feature_names = train_svm(X, y)
#
#     # Generate SHAP explanations
#     generate_shap_explanations(best_svm, X_train_pca, X_test_pca, pca_feature_names)



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
from sklearn.metrics import roc_auc_score, roc_curve

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

    # Preserve PCA component names
    pca_feature_names = [f"PCA_{i+1}" for i in range(pca.n_components_)]

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
    grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_svm = grid_search.best_estimator_
    logging.info(f"Best SVM Parameters: {grid_search.best_params_}")

    # Cross-validation on the full dataset to check for stability
    cv_scores = cross_val_score(best_svm, X_res, y_res, cv=10, scoring='accuracy')
    logging.info(f"Cross-validation scores on full dataset: {cv_scores}")
    logging.info(f"Mean cross-validation accuracy: {np.mean(cv_scores)}")

    # Evaluate the model
    y_pred = best_svm.predict(X_test)
    y_proba = best_svm.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    logging.info(f"SVM Test Accuracy: {accuracy}")
    logging.info(f"AUC-ROC: {auc}")

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"SVM (AUC = {auc:.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(SAVE_DIR, "svm_roc_curve.png"))
    plt.show()

    # Save the model and transformers
    joblib.dump(imputer, os.path.join(SAVE_DIR, "svm_imputer.pkl"))
    joblib.dump(scaler, os.path.join(SAVE_DIR, "svm_scaler.pkl"))
    joblib.dump(pca, os.path.join(SAVE_DIR, "svm_pca.pkl"))
    joblib.dump(best_svm, os.path.join(SAVE_DIR, "best_svm_model.pkl"))
    logging.info("Model and transformers saved successfully.")

    return best_svm, X_train, X_test, pca_feature_names, accuracy
#
#
# def generate_shap_explanations(best_svm, X_train, X_test, pca_feature_names):
#     logging.info("Generating SHAP explanations for SVM model...")
#
#     # Ensure X_train and X_test are numpy arrays for KernelExplainer
#     background_data = shap.sample(X_train, 100, random_state=0)  # Reduced background dataset for efficiency
#     explainer = shap.KernelExplainer(best_svm.predict_proba, background_data)
#
#     # Compute SHAP values for the positive class (index 1)
#     shap_values = explainer.shap_values(X_test[:50])  # Subset for efficiency
#
#     # Check if shap_values is a list (for multi-class)
#     if isinstance(shap_values, list):
#         shap_values = shap_values[1]  # Use SHAP values for the positive class
#
#     # Generate the summary plot
#     plt.figure()
#     shap.summary_plot(shap_values, X_test[:50], feature_names=pca_feature_names, show=False)
#     plt.title("SHAP Summary Plot - SVM")
#     plt.savefig(os.path.join(SAVE_DIR, 'svm_shap_summary_plot.png'), dpi=300, bbox_inches='tight')
#     plt.show()

def generate_shap_explanations(best_svm, X_train, X_test, pca_feature_names):
    logging.info("Generating SHAP explanations for SVM model...")

    # Ensure PCA-transformed data is used
    background_data = shap.sample(X_train, 100, random_state=0)  # Reduced background dataset for efficiency
    explainer = shap.KernelExplainer(best_svm.decision_function, background_data)

    # Compute SHAP values
    shap_values = explainer.shap_values(X_test[:50])  # Subset for explanation

    # Check if feature_names align with PCA features
    if isinstance(pca_feature_names, list) and len(pca_feature_names) == X_test.shape[1]:
        feature_names = pca_feature_names
    else:
        feature_names = [f"PCA_{i}" for i in range(X_test.shape[1])]

    # Generate the summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_test[:50], feature_names=feature_names, show=False)
    # plt.title("SHAP Summary Plot - SVM")
    plt.savefig(os.path.join(SAVE_DIR, 'svm_shap_summary_plot.png'), dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Load and preprocess data
    phenotypic_data, anat_qap, dti_qap, functional_qap = load_data()
    X, y = preprocess_data(phenotypic_data, anat_qap, dti_qap, functional_qap)

    # Train and evaluate SVM
    best_svm, X_train, X_test, pca_feature_names, accuracy = train_svm(X, y)

    # Generate SHAP explanations
    generate_shap_explanations(best_svm, X_train, X_test, pca_feature_names)
