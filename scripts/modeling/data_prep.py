import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Union, Optional
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from .utils import setup_logger
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Configure logger
logger = setup_logger()

def get_categorical_features(X):
    """
    Identify categorical features in dataset.
    
    Args:
        X (pd.DataFrame): Feature matrix
        
    Returns:
        list: List of categorical column names
    """
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Filter out columns that are actually numeric but stored as strings
    numeric_categorical = []
    for col in categorical_cols:
        try:
            # Try to convert to numeric
            pd.to_numeric(X[col], errors='raise')
            numeric_categorical.append(col)
        except (ValueError, TypeError):
            # If conversion fails, it's truly categorical
            pass
    
    # Remove numeric columns that are stored as strings
    true_categorical = [col for col in categorical_cols if col not in numeric_categorical]
    
    logger.info(f"Found {len(true_categorical)} categorical features: {true_categorical}")
    return true_categorical


def prepare_data(
    data: pd.DataFrame,
    target_column: str,
    val_size: float = 0.1,
    random_state: int = 42,
    split_mode: str = "single",
    k_folds: int = 5,
    resampling: Optional[str] = None,
    return_class_weights: bool = True
) -> Union[
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Optional[Dict[int, float]]],
    List[Dict[str, Union[pd.DataFrame, pd.Series, Dict[int, float]]]]
]:
    """
    Prepares the data for training models.

    This function applies various preprocessing steps to the input data:
    1. Splits the dataset into features (X) and target variable (y).
    2. Applies resampling (SMOTE, undersample, or oversample) if specified.
    3. Computes class weights if requested.
    4. Splits the data into training and validation sets based on the selected split mode.

    Args:
        data (pd.DataFrame): The input dataset.
        target_column (str): The name of the column containing the target variable.
        val_size (float, optional): The proportion of the data to use for validation. Defaults to 0.1.
        random_state (int, optional): The seed used for random number generation. Defaults to 42.
        split_mode (str, optional): The mode for splitting the data; can be "single" or "kfold". Defaults to "single".
        k_folds (int, optional): The number of folds in the k-fold cross-validation. Defaults to 5.
        resampling (Optional[str], optional): The method to use for resampling; can be "smote", "undersample", or "oversample". Defaults to None.
        return_class_weights (bool, optional): Whether to compute class weights. Defaults to True.

    Returns:
        Union[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Optional[Dict[int, float]]], List[Dict[str, Union[pd.DataFrame, pd.Series, Dict[int, float]]]]]: 
        The preprocessed data in the format of either a tuple (X_train, y_train, class_weights) or a list of dictionaries (folds).
    """


    # split
    X = data.drop(columns = [target_column, "TransactionID"])
    y = data[target_column]

    def apply_resampling(X, y):
        """
        Applies resampling to the data if specified.

        Args:
            X (pd.DataFrame): The feature matrix.
            y (pd.Series): The target variable.

        Returns:
            pd.DataFrame, pd.Series: The resampled features and target variable.
        """

        if resampling == "smote":
            sampler = SMOTE(random_state = random_state)
        elif resampling == "undersample":
            sampler = RandomUnderSampler(random_state = random_state)
        elif resampling == "oversample":
            sampler = RandomOverSampler(random_state = random_state)
        else:
            return X, y

        X_res, y_res = sampler.fit_resample(X, y)
        return X_res, y_res

    def get_class_weight(y):
        """
        Computes class weights for the target variable.

        Args:
            y (pd.Series): The target variable.

        Returns:
            Dict[int, float]: A dictionary of class weights.
        """
        
        classes = np.unique(y)
        weights = compute_class_weight(class_weight = "balanced", classes = classes, y = y)

        return {cls: w for cls, w in zip(classes, weights)}

    # split modes
    if split_mode == "single":
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size = val_size, random_state = random_state, stratify = y
        )

        X_train, y_train = apply_resampling(X_train, y_train)
        class_weights = get_class_weight(y_train) if return_class_weights else None
        
        
        return X_train, y_train, class_weights

    elif split_mode == "kfold":
        skf = StratifiedKFold(n_splits = k_folds, shuffle = True, random_state = random_state)
        folds = []
        for train_idx, val_idx in skf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            X_train, y_train = apply_resampling(X_train, y_train)
            class_weights = get_class_weight(y_train) if return_class_weights else None

            folds.append({
                "X_train": X_train,
                "y_train": y_train,
                "X_val": X_val,
                "y_val": y_val,
                "class_weights": class_weights
            })
        return folds

    else:
        raise ValueError("split_mode must be either 'single' or 'kfold'")
















