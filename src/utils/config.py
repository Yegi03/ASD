# src/utils/config.py
import os

# Base directory of the project
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Paths to data files
DATA_PATHS = {
    "phenotypic": os.path.join(BASE_DIR, "data", "ABIDEII_Composite_Phenotypic.csv"),
    "anat_qap": os.path.join(BASE_DIR, "data", "ABIDEII_MRI_Quality_Metrics", "anat_qap.csv"),
    "dti_qap": os.path.join(BASE_DIR, "data", "ABIDEII_MRI_Quality_Metrics", "dti_qap.csv"),
    "functional_qap": os.path.join(BASE_DIR, "data", "ABIDEII_MRI_Quality_Metrics", "functional_qap.csv")
}

# Save directory for models and transformers
SAVE_DIR = os.path.join(BASE_DIR, "models", "saved_models")
os.makedirs(SAVE_DIR, exist_ok=True)

# Shared training configuration
TRAIN_TEST_SPLIT = {
    "test_size": 0.2,
    "random_state": 42
}

# Model-specific parameters
MODEL_PARAMS = {
    "gbm": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "min_samples_split": [2, 5, 10],
        "random_state": 42
    },
    "xgboost": {
        "param_grid": {
            "xgb__learning_rate": [0.01, 0.05, 0.1],
            "xgb__max_depth": [3, 5, 7],
            "xgb__n_estimators": [50, 100, 200]
        },
        "default_params": {
            "colsample_bytree": 0.3,
            "gamma": 0.2,
            "reg_alpha": 1.5,
            "reg_lambda": 1.5,
            "subsample": 0.6
        }

    },
    "svm": {
        "C": [1, 10, 100],
        "gamma": ["scale", "auto"],
        "kernel": ["rbf"]
    }
}

# Target column label in data
TARGET_LABEL = "DX_GROUP"
