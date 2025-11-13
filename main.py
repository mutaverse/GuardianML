import time
import pandas as pd
from scripts.data_loading import load_dataset
from scripts.modeling.data_prep import prepare_data
from scripts.wranglers import FraudDataWrangler, FraudDatasetMerger
from scripts.modeling.orchestrator import Orchestrator

start_time = time.perf_counter()

# Load datasets
transaction_df = load_dataset("dataset/train_transaction.csv")
identity_df = load_dataset("dataset/train_identity.csv")

# Handle preprocessing
transaction_df = FraudDataWrangler(dataset_type = "transaction").fit_transform(transaction_df)
identity_df = FraudDataWrangler(dataset_type = "identity").fit_transform(identity_df)

# Merge datasets
merged_df = FraudDatasetMerger(joint_type = "left").fit_transform((transaction_df, identity_df))

# Prepare data (for training)
split_mode = "kfold"  # Change to "kfold" for k-fold cross validation mode

if split_mode == "single":
    X, y, class_weights = prepare_data(merged_df, "isFraud", split_mode="single")
    
    # Print class weights
    print("Class weights:", class_weights)
    
    # Print data shapes
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # train model
    orchestrator = Orchestrator(models = ["xgb", "lgbm"], tuning = "random")
    orchestrator.run(X, y, None, None, class_weights)  # No validation split for single mode
    
elif split_mode == "kfold":
    folds = prepare_data(merged_df, "isFraud", split_mode="kfold", k_folds = 2)
    
    print(f"Number of folds: {len(folds)}")
    
    # Process each fold
    for i, fold in enumerate(folds):
        X_train, y_train = fold["X_train"], fold["y_train"]
        X_val, y_val = fold["X_val"], fold["y_val"]
        class_weights = fold["class_weights"]
        
        print(f"\nFold {i+1}:")
        print(f"  Class weights: {class_weights}")
        print(f"  X_train shape: {X_train.shape}")
        print(f"  y_train shape: {y_train.shape}")
        print(f"  X_val shape: {X_val.shape}")
        print(f"  y_val shape: {y_val.shape}")
        
        # train model for this fold
        orchestrator = Orchestrator(models = ["xgb"], tuning = "optuna")
        orchestrator.run(X_train, y_train, X_val, y_val, class_weights)

print(f"Total time taken: {(time.perf_counter() - start_time):.2f}s")
