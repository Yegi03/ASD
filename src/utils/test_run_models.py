# import joblib
# import os
# import logging
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from src.data_preprocessing.data_preprocessing import preprocess_data
# from src.utils.config import DATA_PATHS, SAVE_DIR, TRAIN_TEST_SPLIT
#
# # Set up logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
#
# def handle_nans_and_infs(X):
#     """Fill NaNs with column means and replace infinities with finite values on a per-column basis."""
#     X_numeric = X.select_dtypes(include=['number'])
#
#     # Fill NaNs in each column with the column mean
#     X_numeric_filled = X_numeric.fillna(X_numeric.mean())
#
#     # Replace infinities in each column with the column mean individually
#     for col in X_numeric_filled.columns:
#         col_mean = X_numeric_filled[col].mean()
#         X_numeric_filled[col] = X_numeric_filled[col].replace([np.inf, -np.inf], col_mean)
#
#     # Combine back with non-numeric data
#     X_non_numeric = X.select_dtypes(exclude=['number'])
#     return pd.concat([X_numeric_filled, X_non_numeric], axis=1)
#
# # Load data with ISO-8859-1 encoding to handle non-UTF-8 characters
# logging.info("Loading datasets for preprocessing.")
# phenotypic_data = pd.read_csv(DATA_PATHS["phenotypic"], encoding="ISO-8859-1")
# anat_qap = pd.read_csv(DATA_PATHS["anat_qap"], encoding="ISO-8859-1")
# dti_qap = pd.read_csv(DATA_PATHS["dti_qap"], encoding="ISO-8859-1")
# functional_qap = pd.read_csv(DATA_PATHS["functional_qap"], encoding="ISO-8859-1")
#
# # Preprocess data
# X, y = preprocess_data(phenotypic_data, anat_qap, dti_qap, functional_qap)
#
# # Handle NaN and infinite values
# X = handle_nans_and_infs(X)
#
# # Scale numeric features
# scaler = StandardScaler()
# X_numeric = X.select_dtypes(include=['number'])
# X_scaled = pd.DataFrame(scaler.fit_transform(X_numeric), columns=X_numeric.columns, index=X_numeric.index)
#
# # Replace numeric columns in X with the scaled values
# X.update(X_scaled)
#
# # Final fillna to handle any remaining NaNs
# X = X.fillna(0)  # Replace any remaining NaNs with 0 (or choose another value as needed)
#
# # Log final check for NaNs
# if X.isna().sum().sum() > 0:
#     logging.warning("Warning: There are still NaN values present after final handling.")
# else:
#     logging.info("All NaN values have been handled.")
#
# # Split data
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=TRAIN_TEST_SPLIT["test_size"], random_state=TRAIN_TEST_SPLIT["random_state"]
# )
#
# # Save consistent test and train data
# test_data_path = os.path.join(SAVE_DIR, "test_data.pkl")
# joblib.dump((X_train, X_test, y_train, y_test), test_data_path)
# logging.info(f"Training and test data saved to {test_data_path}.")


import joblib
import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.data_preprocessing.data_preprocessing import preprocess_data
from src.utils.config import DATA_PATHS, SAVE_DIR, TRAIN_TEST_SPLIT

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def handle_nans_and_infs(X):
    """Fill NaNs with column means and replace infinities with finite values on a per-column basis."""
    X_numeric = X.select_dtypes(include=['number'])
    X_numeric_filled = X_numeric.fillna(X_numeric.mean())

    for col in X_numeric_filled.columns:
        col_mean = X_numeric_filled[col].mean()
        X_numeric_filled[col] = X_numeric_filled[col].replace([np.inf, -np.inf], col_mean)

    X_non_numeric = X.select_dtypes(exclude=['number'])
    return pd.concat([X_numeric_filled, X_non_numeric], axis=1)


def main():
    logging.info("Loading datasets for preprocessing.")

    try:
        phenotypic_data = pd.read_csv(DATA_PATHS["phenotypic"], encoding="ISO-8859-1")
        anat_qap = pd.read_csv(DATA_PATHS["anat_qap"], encoding="ISO-8859-1")
        dti_qap = pd.read_csv(DATA_PATHS["dti_qap"], encoding="ISO-8859-1")
        functional_qap = pd.read_csv(DATA_PATHS["functional_qap"], encoding="ISO-8859-1")
    except Exception as e:
        logging.error(f"Error loading data files: {e}")
        return

    X, y = preprocess_data(phenotypic_data, anat_qap, dti_qap, functional_qap)

    X = handle_nans_and_infs(X)

    scaler = StandardScaler()
    X_numeric = X.select_dtypes(include=['number'])
    X_scaled = pd.DataFrame(scaler.fit_transform(X_numeric), columns=X_numeric.columns, index=X_numeric.index)
    X.update(X_scaled)

    X = X.fillna(0)  # Replace any remaining NaNs with 0

    if X.isna().sum().sum() > 0:
        logging.warning("Warning: There are still NaN values present after final handling.")
    else:
        logging.info("All NaN values have been handled.")

    for state in [42, 24, 101, 2021]:  # Different random states
        logging.info(f"Evaluating with random_state = {state}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TRAIN_TEST_SPLIT["test_size"], random_state=state
        )

        # Proceed with training and evaluation here...



    test_data_path = os.path.join(SAVE_DIR, "test_data.pkl")

    try:
        joblib.dump((X_train, X_test, y_train, y_test), test_data_path)
        logging.info(f"Training and test data saved to {test_data_path}.")
    except Exception as e:
        logging.error(f"Error saving data: {e}")


if __name__ == "__main__":
    main()
