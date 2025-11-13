import pandas as pd
import numpy as np
import time
from scripts.modeling.utils import setup_logger

# Configure logger
logger = setup_logger()



# for transaction dataset
def trans_feature_engineering(dataset, rare_provider_threshold=500):
    """
    Perform feature engineering operations on transaction dataset.

    This function applies a set of rules to the dataset to create new features:
    1. Combines billing address columns into a single "billing_address" feature.
    2. Extracts provider information from email addresses and creates corresponding features ("provider", "suffix").
    3. Categorizes providers based on their count in the dataset, using a threshold for rare providers.
    4. Creates a new feature based on fraud statistics per domain.

    Args:
        dataset (pd.DataFrame): The input dataset to process.
        rare_provider_threshold (int, optional): The minimum number of providers required 
        for a category to be considered "rare". Defaults to 500.

    Returns:
        pd.DataFrame: The processed dataset with new features added.
    """

    start_time = time.perf_counter()
    new_features = pd.DataFrame(index=dataset.index)

    # billing address
    if "addr1" in dataset.columns and "addr2" in dataset.columns:
        new_features["billing_address"] = dataset["addr1"].astype(str) + "_" + dataset["addr2"].astype(str)

    # email provider features
    if "P_emaildomain" in dataset.columns:
        new_features["provider"] = dataset["P_emaildomain"].apply(
            lambda x: x.split(".")[0] if x != "missing" else "missing"
        )
        new_features["suffix"] = dataset["P_emaildomain"].apply(
            lambda x: x.split(".")[-1] if x != "missing" else "missing"
        )

        provider_counts = new_features["provider"].value_counts()
        new_features["provider_bucket"] = new_features["provider"].apply(
            lambda x: x if provider_counts[x] >= rare_provider_threshold else "rare"
        )

    dataset = pd.concat([dataset, new_features], axis=1)

    # fraud bucket per domain
    if "P_emaildomain" in dataset.columns and "isFraud" in dataset.columns:
        fraud_stats = dataset.groupby("P_emaildomain")["isFraud"].agg(["mean", "count"]).reset_index()
        fraud_stats["fraud_bucket"] = pd.cut(
            fraud_stats["mean"], bins=[-1, 0.03, 0.10, 1], labels=["low", "medium", "high"]
        ).astype(str)

        dataset = dataset.merge(
            fraud_stats[["P_emaildomain", "fraud_bucket"]],
            on="P_emaildomain", how="left"
        )

        dataset.drop(columns=["P_emaildomain"], inplace=True)
    
    logger.info(f"Feature engineering completed. Time taken: {(time.perf_counter() - start_time):.2f}s")
    return dataset





# Function for feature engineering
def identity_feature_engineering(dataset, rare_device_info_threshold=200):
    """
    Feature engineering for identity dataset:
      - Bucket rare DeviceInfo values into 'others'
      - Regroup/simplify DeviceInfo categories into broader families

    Args:
        dataset : pd.DataFrame
            The identity dataset.
        rare_device_info_threshold : int, default=200
            Minimum frequency for a DeviceInfo to be kept. 
            Rare categories are replaced with 'others'.

    Returns:
        pd.DataFrame
            Dataset with engineered DeviceInfo.
    """
    
    start_time = time.perf_counter()
    df = dataset.copy()

    # --- Step 1: Handle rare DeviceInfo ---
    if 'DeviceInfo' in df.columns:
        device_counts = df['DeviceInfo'].value_counts()
        frequent_devices = device_counts[device_counts >= rare_device_info_threshold].index
        df['DeviceInfo'] = df['DeviceInfo'].where(df['DeviceInfo'].isin(frequent_devices), 'others')
    
        logger.info(
            f"DeviceInfo engineered: {len(frequent_devices)} frequent categories kept, "
            f"others bucketed into 'others'."
        )
        # --- Step 2: Vectorized regrouping ---
        device_series = df['DeviceInfo'].astype(str)
    
        conditions = [
            device_series.str.startswith('rv:'),                         # Firefox
            device_series.str.startswith('SM-'),                         # Samsung Galaxy
            device_series.str.startswith('Moto'),                        # Motorola
            device_series.str.contains('Trident/7.0|rv:11.0', regex=True), # IE 11
            device_series.str.contains('Windows', regex=False),          # Windows
            device_series.str.contains('iOS', regex=False),              # iOS Device
            device_series.str.contains('MacOS', regex=False),            # MacOS
            device_series.eq('SAMSUNG'),                                 # Exact Samsung string
        ]
        choices = [
            'Firefox',
            'Samsung',
            'Motorola',
            'Internet Explorer',
            'Windows',
            'iOS Device',
            'MacOS',
            'Samsung'
        ]
    
        df['DeviceInfo'] = np.select(conditions, choices, default=device_series)

    logger.info(f"DeviceInfo categories simplified. Unique categories: {df['DeviceInfo'].nunique()}")
    
    logger.info(f"Feature Engineering completed. | Time taken: {(time.perf_counter() - start_time):.2f}s")
    return df