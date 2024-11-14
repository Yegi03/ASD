import pandas as pd

# File paths
file_paths = {
    "anat_qap": "ABIDEII_MRI_Quality_Metrics/anat_qap.csv",
    "dti_qap": "ABIDEII_MRI_Quality_Metrics/dti_qap.csv",
    "functional_qap": "ABIDEII_MRI_Quality_Metrics/functional_qap.csv",
    "phenotypic_data": "ABIDEII_Composite_Phenotypic.csv"
}

# Load the data with a specified encoding (ISO-8859-1 or cp1252)
try:
    anat_qap = pd.read_csv(file_paths["anat_qap"], encoding="ISO-8859-1")
    dti_qap = pd.read_csv(file_paths["dti_qap"], encoding="ISO-8859-1")
    functional_qap = pd.read_csv(file_paths["functional_qap"], encoding="ISO-8859-1")
    phenotypic_data = pd.read_csv(file_paths["phenotypic_data"], encoding="ISO-8859-1")
except UnicodeDecodeError:
    print("Error decoding one of the CSV files. Trying with different encoding.")
    anat_qap = pd.read_csv(file_paths["anat_qap"], encoding="cp1252")
    dti_qap = pd.read_csv(file_paths["dti_qap"], encoding="cp1252")
    functional_qap = pd.read_csv(file_paths["functional_qap"], encoding="cp1252")
    phenotypic_data = pd.read_csv(file_paths["phenotypic_data"], encoding="cp1252")

# Explore each dataset
def explore_data(df, name):
    print(f"--- {name} ---")
    print(f"Shape of the data: {df.shape}")
    print(f"Column names: {df.columns.tolist()}")
    print("\nFirst few rows of the dataset:")
    print(df.head())
    print("\nData types and null values:")
    print(df.info())
    print("\nStatistical summary of numerical data:")
    print(df.describe())
    print("\nMissing values:")
    print(df.isnull().sum())
    print("="*50)

# Explore all datasets
explore_data(anat_qap, "Anatomical Quality Assurance (anat_qap)")
explore_data(dti_qap, "DTI Quality Assurance (dti_qap)")
explore_data(functional_qap, "Functional Quality Assurance (functional_qap)")
explore_data(phenotypic_data, "Phenotypic Data (ABIDEII_Composite_Phenotypic)")
