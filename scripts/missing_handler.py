import time
from scripts.modeling.utils import setup_logger

# Configure logger
logger = setup_logger()


# Function for transaction dataset
def handle_transaction_missing(dataset, drop_threshold=0.7, impute_categoricals=True, impute_numericals="median"):
    """
    Drop and impute missing values based on rules (Transaction dataset).

    This function applies a set of rules to a given transaction dataset:
    1. Drops features with more than the specified threshold of missingness.
    2. Imputes categorical variables with any missing values using either "missing" or -1 placeholder values.
    3. Imputes numerical variables based on the specified method (mean or median).

    Args:
        dataset (pd.DataFrame): The input dataset to process (transaction dataset).
        drop_threshold (float, optional): The minimum proportion of missing values required for a feature to be dropped. Defaults to 0.7.
        impute_categoricals (bool, optional): Whether to impute categorical variables. Defaults to True.
        impute_numericals (str or None, optional): Method to use for imputing numerical variables; 
        can be "mean", "median", or None (imputation not performed). Defaults to "median".

    Returns:
        pd.DataFrame: The processed dataset with missing values addressed.

    Raises:
        ValueError: If an invalid value is specified for impute_numericals.   
    """
    
    missing = dataset.isna().mean()
    to_drop = missing[missing >= drop_threshold].index

    if len(to_drop) > 0:
        dataset = dataset.drop(columns=to_drop)
        logger.info(f"Dropped {len(to_drop)} feature(s) with >={drop_threshold:.0%} missingness.")

    # impute categoricals
    if impute_categoricals:
        categorical_cols = dataset.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            if dataset[col].isna().any():
                dataset[col] = dataset[col].fillna("missing")

        for col in ["card1", "card2", "card3", "card5", "addr1", "addr2"]:
            if col in dataset.columns and dataset[col].isna().any():
                dataset[col] = dataset[col].fillna(-1)

    # impute numericals
    if impute_numericals is not None:
        num_cols = dataset.select_dtypes(include="number").columns
        for col in num_cols:
            if dataset[col].isna().any():
                if impute_numericals == "mean":
                    fill_value = dataset[col].mean()
                elif impute_numericals == "median":
                    fill_value = dataset[col].median()
                else:
                    raise ValueError("Invalid impute_numericals option. Use 'mean', 'median', or None.")
                dataset[col] = dataset[col].fillna(fill_value)

    return dataset


# Function for identity dataset
def handle_identity_missing(
    dataset,
    drop_threshold=0.7,
    impute_categoricals=True,
    impute_numericals="median"
):
    """
    Drop and impute missing values based on rules (Identity dataset).

    This function applies a set of rules to a given identity dataset to address missing values:
    1. Drops features with more than the specified threshold of missingness.
    2. Imputes categorical variables with any missing values using "missing" placeholder values.
    3. Imputes numerical variables based on the specified method (mean or median).

    Args:
        dataset (pd.DataFrame): The input dataset to process.
        drop_threshold (float, optional): The minimum proportion of missing values required for a feature to be dropped. Defaults to 0.7.
        impute_categoricals (bool, optional): Whether to impute categorical variables. Defaults to True.
        impute_numericals (str or None, optional): Method to use for imputing numerical variables; 
        can be "mean", "median", or None (imputation not performed). Defaults to "median".

    Returns:
        pd.DataFrame: The processed dataset with missing values addressed.
    """
    start_time = time.perf_counter()
    missing = dataset.isna().mean()
    to_drop = missing[missing >= drop_threshold].index

    if len(to_drop) > 0:
        dataset = dataset.drop(columns=to_drop)
        logger.info(f"Dropped {len(to_drop)} feature(s) with >={drop_threshold:.0%} missingness.")

    # impute categoricals
    if impute_categoricals:
        categorical_cols = dataset.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            if dataset[col].isna().any():
                dataset[col] = dataset[col].fillna("missing")
        logger.info("Imputed categorical missing values")

    # impute numericals
    if impute_numericals is not None:
        num_cols = dataset.select_dtypes(include="number").columns
        for col in num_cols:
            if dataset[col].isna().any():
                if impute_numericals == "mean":
                    fill_value = dataset[col].mean()
                elif impute_numericals == "median":
                    fill_value = dataset[col].median()
                else:
                    raise ValueError("Invalid impute_numericals option. Use 'mean', 'median', or None.")
                dataset[col] = dataset[col].fillna(fill_value)
        logger.info("Imputed numerical values")

    logger.info(f"Data cleaning completed | Time taken: {(time.perf_counter() - start_time):.2f}s")
    return dataset
