# src/data_preprocessing/feature_engineering.py

from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import pandas as pd
import logging

def create_features(X, y, apply_pca=False, n_components=50):
    logging.info("Starting feature engineering...")

    # Select only numeric columns
    X_numeric = X.select_dtypes(include=[float, int])

    # Apply Variance Threshold
    selector = VarianceThreshold(threshold=0.01)
    X_selected = selector.fit_transform(X_numeric)
    selected_features = X_numeric.columns[selector.get_support(indices=True)]
    X_selected_df = pd.DataFrame(X_selected, columns=selected_features)
    logging.info(f"Selected features shape after variance threshold: {X_selected_df.shape}")

    # Impute missing values
    imputer = SimpleImputer(strategy="mean")
    X_imputed_df = pd.DataFrame(imputer.fit_transform(X_selected_df), columns=selected_features)

    # Apply PCA if specified
    if apply_pca:
        logging.info(f"Applying PCA with {n_components} components...")
        pca = PCA(n_components=n_components)
        X_pca_df = pd.DataFrame(pca.fit_transform(X_imputed_df), columns=[f"PC{i+1}" for i in range(n_components)])
        logging.info(f"PCA applied. Shape after PCA: {X_pca_df.shape}")
        return X_pca_df, y

    return X_imputed_df, y
