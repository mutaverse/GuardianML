from scripts.modeling.utils import setup_logger

# Configure logger
logger = setup_logger()

# Function to handle duplicates in transaction datasets
def handle_duplicates(dataset, id_col="TransactionID"):
    """
    Drop duplicate rows based on an ID column.

    This function applies a simple rule to a given dataset to address duplicate rows:
    1. Checks if the specified ID column exists in the dataset.
    2. Drops all duplicate rows based on that ID column, keeping only the first occurrence of each ID value.
    3. Logs the number of dropped duplicate rows (or indicates no duplicates found).

    Args:
        dataset (pd.DataFrame): The input dataset to process.
        id_col (str, optional): The name of the column containing the IDs to use for dropping duplicates. Defaults to "TransactionID".

    Returns:
        pd.DataFrame: The processed dataset with duplicate rows addressed.
    """
    if id_col in dataset.columns:
        before = dataset.shape[0]
        dataset = dataset.drop_duplicates(subset=[id_col], keep="first")
        dropped = before - dataset.shape[0]
        if dropped > 0:
            logger.info(f"Dropped {dropped} duplicate rows.")
        else:
            logger.info("No duplicates found!")
    return dataset