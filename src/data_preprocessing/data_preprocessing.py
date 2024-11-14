import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from src.utils.config import TARGET_LABEL  # Assuming you add TARGET_LABEL to config

def preprocess_data(phenotypic_data, anat_qap, dti_qap, functional_qap):
    logging.info("Preprocessing data...")

    # Standardize column names
    rename_columns(phenotypic_data, anat_qap, dti_qap, functional_qap)

    # Merge data
    merged_data = phenotypic_data.merge(anat_qap, on=['SUB_ID', 'SITE_ID'], how='inner') \
        .merge(dti_qap, on=['SUB_ID', 'SITE_ID'], how='inner') \
        .merge(functional_qap, on=['SUB_ID', 'SITE_ID'], how='inner')
    logging.info(f"Merged data shape: {merged_data.shape}")

    # Define target label and separate X, y
    if TARGET_LABEL in merged_data.columns:
        merged_data = merged_data.dropna(subset=[TARGET_LABEL])
        X = merged_data.drop(columns=[TARGET_LABEL])
        y = merged_data[TARGET_LABEL]
        logging.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    else:
        raise ValueError(f"Target label '{TARGET_LABEL}' not found in the merged data.")

    # Fill NaNs in numeric data
    X = fill_nans(X)

    # Log numeric columns
    X_numeric = X.select_dtypes(include=['number'])
    logging.info(f"Columns in X_numeric: {X_numeric.columns.tolist()}")

    # Outlier removal
    columns_to_check = [col for col in ['AGE_AT_SCAN', 'CNR', 'SNR'] if col in X_numeric.columns]
    X = remove_outliers(X, columns_to_check=columns_to_check, z_threshold=4)
    logging.info(f"Data shape after selective outlier removal: {X.shape}")

    # Scale numeric features
    X = scale_features(X)

    return X, y

def rename_columns(*dfs):
    """Renames columns across dataframes to standardize merging."""
    for df in dfs:
        df.rename(columns={'Sub_ID': 'SUB_ID', 'Site_ID': 'SITE_ID'}, inplace=True)


def fill_nans(X):
    """Fill NaNs in numeric columns with column means and handle infinities."""
    X_numeric = X.select_dtypes(include=['number'])

    # Fill NaN values with the mean of each column
    X_numeric_filled = X_numeric.fillna(X_numeric.mean())

    # Replace any infinities with the mean of each respective column
    for col in X_numeric_filled.columns:
        col_mean = X_numeric_filled[col].mean()
        X_numeric_filled[col] = X_numeric_filled[col].replace([float('inf'), -float('inf')], col_mean)

    # Handle non-numeric columns separately
    X_non_numeric = X.select_dtypes(exclude=['number'])

    return pd.concat([X_numeric_filled, X_non_numeric], axis=1)


def remove_outliers(X, columns_to_check=None, z_threshold=4):
    """Remove rows with outliers based on specific columns and z-score threshold."""
    X_numeric = X.select_dtypes(include=['number'])
    initial_shape = X.shape[0]

    for col in columns_to_check:
        col_z_scores = ((X_numeric[col] - X_numeric[col].mean()) / X_numeric[col].std()).abs()
        rows_with_outliers = (col_z_scores > z_threshold).sum()
        logging.info(f"Outliers in {col}: {rows_with_outliers} rows flagged.")
        X = X[col_z_scores <= z_threshold]

    rows_removed = initial_shape - X.shape[0]
    logging.info(f"Total rows removed due to outliers: {rows_removed}")
    return X

def scale_features(X):
    """Standardize numeric features."""
    X_numeric = X.select_dtypes(include=['number'])
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_numeric), columns=X_numeric.columns)
    X = pd.concat([X_scaled, X.select_dtypes(exclude=['number'])], axis=1)
    logging.info("Numeric features scaled.")
    return X
