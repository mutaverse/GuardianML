import pandas as pd
import time
from scripts.modeling.utils import setup_logger

# Configure logger
logger = setup_logger()

def encode_categorical_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Encode all categorical (object/category) columns in the dataset using integer factorization.

    Args:
        dataset (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Dataset with categorical features encoded as integers.
    """
    start = time.perf_counter()

    # handle duplicate columns
    if dataset.columns.duplicated().any():
        dup_cols = dataset.columns[dataset.columns.duplicated()].unique().tolist()
        logger.warning(f"Duplicate columns detected and will be deduplicated: {dup_cols}")
        dataset = dataset.loc[:, ~dataset.columns.duplicated()]

    cat_cols = dataset.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols:
        logger.info("No categorical feature found for encoding.")
        return dataset
        
    for col in cat_cols:
        col_series = dataset[col]
        codes, uniques = pd.factorize(col_series)
        dataset[col] = codes
        logger.info(f"Encoded `{col}` with {len(uniques)} unique values.")

        
    logger.info(f"Encoding categorical columns completed. | Time taken: {time.perf_counter() - start:.2f}s")
    return dataset