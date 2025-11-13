import os
import pandas as pd

# Function to load dataset
def load_dataset(path_to_dataset):
    """Load dataset from various file types."""
    if not os.path.exists(path_to_dataset):
        raise FileNotFoundError(f"File not found: {path_to_dataset}")

    ext = os.path.splitext(path_to_dataset)[1].lower()
    loaders = {
        ".csv": pd.read_csv,
        ".xlsx": pd.read_excel,
        ".xls": pd.read_excel,
        ".parquet": pd.read_parquet,
        ".json": pd.read_json,
    }

    if ext not in loaders:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {list(loaders.keys())}")

    return loaders[ext](path_to_dataset)