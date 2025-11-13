import time
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from .duplicates_handler import handle_duplicates
from .feature_engineering import trans_feature_engineering, identity_feature_engineering
from .missing_handler import handle_transaction_missing, handle_identity_missing
from .encoding_handler import encode_categorical_features
from scripts.modeling.utils import setup_logger


# Configure logger
logger = setup_logger()


# Registry of dataset-specific configs
WRANGLER_CONFIG = {
    "transaction": {
        "missing_handler": handle_transaction_missing,
        "feature_engineering": trans_feature_engineering,
        "threshold_param": "rare_provider_threshold",
        "extra_steps": lambda df: df.assign(TransactionAmt=df["TransactionAmt"].round(2)) if "TransactionAmt" in df else df
    },
    "identity": {
        "missing_handler": handle_identity_missing,
        "feature_engineering": identity_feature_engineering,
        "threshold_param": "rare_device_info_threshold",
        "extra_steps": lambda df: df  # no special extra step for now
    }
}

# --------- For Data Wrangling -----------------

class FraudDataWrangler(BaseEstimator, TransformerMixin):
    """
        A generic data wrangler for transaction and identity datasets.

        This class is designed to work as an sklearn transformer. It provides a way to 
        apply various preprocessing steps to the input dataset.
    """
    
    def __init__(
        self,
        dataset_type = "transaction",
        drop_threshold = 0.7,
        impute_categoricals = True,
        impute_numericals = "median",
        feature_engineering_flag = True,
        rare_provider_threshold = 500,
        rare_device_info_threshold = 200
    ):
        """
        Initializes the FraudDataWrangler with various configuration options.

        Args:
            dataset_type (str): The type of dataset to process. Can be 'transaction' or 'identity'.
            drop_threshold (float): The minimum proportion of missing values required for a feature to be dropped.
            impute_categoricals (bool): Whether to impute categorical variables.
            impute_numericals (str): Method to use for imputing numerical variables; can be "mean", "median", or None (imputation not performed).
            feature_engineering_flag (bool): Whether to apply feature engineering steps.
            rare_provider_threshold (int): The minimum number of providers required for a category to be considered "rare".
            rare_device_info_threshold (int): The minimum number of device info values required for a category to be considered "rare".
        """

        self.dataset_type = dataset_type
        self.drop_threshold = drop_threshold
        self.impute_categoricals = impute_categoricals
        self.impute_numericals = impute_numericals
        self.feature_engineering_flag = feature_engineering_flag
        self.rare_provider_threshold = rare_provider_threshold
        self.rare_device_info_threshold = rare_device_info_threshold

    def fit(self, X, y=None):
        """
        Fits the model without updating any parameters.

        This method is used to initialize the transformer. It does not perform any learning.
        """
        return self


    def transform(self, X):
        """
        Applies the preprocessing steps to the input data.

        Args:
            X (pd.DataFrame): The input dataset.

        Returns:
            pd.DataFrame: The preprocessed dataset.
        """

        start_time = time.perf_counter()
        cfg = WRANGLER_CONFIG[self.dataset_type]

        # 1. Remove duplicates
        X = handle_duplicates(X)

        # 2. Handle missing values
        X = cfg["missing_handler"](X,
                                  self.drop_threshold,
                                  self.impute_categoricals,
                                  self.impute_numericals)

        # 3. Dataset-specific extra steps
        X = cfg["extra_steps"](X)

        # 4. Feature engineering
        if self.feature_engineering_flag:
            threshold_value = (self.rare_provider_threshold
                              if self.dataset_type == "transaction"
                              else self.rare_device_info_threshold)
            X = cfg["feature_engineering"](X, threshold_value)

        # 5. Encode categorical features
        X = encode_categorical_features(X)

        elapsed_time = time.perf_counter() - start_time
        logger.info(f"[{self.dataset_type}] transform() complete | Shape: {X.shape} | Time: {elapsed_time:.2f}s")
        return X



# ------------ For merging cleaned datasets ----------------

class FraudDatasetMerger(BaseEstimator, TransformerMixin):
    """"
    A wrapper to merge cleaned transaction and identity datasets.
    """
    def __init__(
        self,
        drop_threshold = 0.7,
        impute_categoricals = True,
        impute_numericals = "median",
        feature_engineering_flag = True,
        rare_provider_threshold = 500,
        rare_device_info_threshold = 200,
        joint_type = "left"
        
    ):
        """
        Initializes the FraudDatasetMerger with various configuration options.

        Args:
            drop_threshold (float): The minimum proportion of missing values required for a feature to be dropped. Default = 0.7
            impute_categoricals (bool): Whether to impute categorical variables.
            impute_numericals (str): Method to use for imputing numerical variables; can be "mean", "median", or None (imputation not performed).
            feature_engineering_flag (bool): Whether to apply feature engineering steps.
            rare_provider_threshold (int): The minimum number of providers required for a category to be considered "rare".
            rare_device_info_threshold (int): The minimum number of device info values required for a category to be considered "rare".
            joint_type (str): The type of merge to use. Can be 'left', 'right', or 'outer'.
        """

        self.drop_threshold = drop_threshold
        self.impute_categoricals = impute_categoricals
        self.impute_numericals = impute_numericals
        self.feature_engineering_flag = feature_engineering_flag
        self.rare_provider_threshold = rare_provider_threshold
        self.rare_device_info_threshold = rare_device_info_threshold
        self.joint_type = joint_type

    def fit(self, X, y = None):
        """
        Fits the model without updating any parameters.

        This method is used to initialize the transformer. It does not perform any learning.
        """
        return self

    def transform(self, datasets):
        """
        Merges two cleaned datasets into one.

        Args:
            datasets (tuple of pd.DataFrame): The transaction and identity datasets to merge.

        Returns:
            pd.DataFrame: The merged dataset.
        """

        transaction_df, identity_df = datasets
        start_time = time.perf_counter()

        # Wrangle each dataset separately
        transaction_wrangler = FraudDataWrangler(
            dataset_type = "transaction",
            drop_threshold = self.drop_threshold,
            impute_numericals = self.impute_numericals,
            impute_categoricals = self.impute_categoricals,
            feature_engineering_flag = self.feature_engineering_flag,
            rare_provider_threshold = self.rare_provider_threshold
        )

        identity_wrangler = FraudDataWrangler(
            dataset_type = "identity",
            drop_threshold = self.drop_threshold,
            impute_numericals = self.impute_numericals,
            impute_categoricals = self.impute_categoricals,
            feature_engineering_flag = self.feature_engineering_flag,
            rare_device_info_threshold = self.rare_device_info_threshold
        )

        transaction_clean = transaction_wrangler.fit_transform(transaction_df)
        identity_clean = identity_wrangler.fit_transform(identity_df)

        # Merge datasets
        merged = pd.merge(
            transaction_clean,
            identity_clean,
            how = self.joint_type,
            on = "TransactionID"
        )

        elapsed_time = time.perf_counter() - start_time
        logger.info(f"Merged dataset | Shape: {merged.shape} | Time: {elapsed_time:.2f}s")

        return merged
        














