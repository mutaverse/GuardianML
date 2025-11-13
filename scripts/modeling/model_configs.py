from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Registry of model configurations
MODEL_CONFIGS = {
    "rf": {
        "class": RandomForestClassifier,
        "baseline": {
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_split": 2,
            "n_jobs": -1,
            "class_weight": "balanced"
        },
        "param_grid": {
            "n_estimators": [100, 200, 500],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
            "max_features": ["sqrt", "log2"]
        },
        "optuna": {
            "n_estimators": {"type": "int", "low": 100, "high": 1000},
            "max_depth": {"type": "int", "low": 5, "high": 50},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
            "max_features": {"type": "categorical", "choices": ["sqrt", "log2"]}
        }
    },

    "xgb": {
        "class": XGBClassifier,
        "baseline": {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "n_jobs": -1
        },
        "param_grid": {
            "n_estimators": [200, 500, 1000],
            "max_depth": [3, 6, 10],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0]
        },
        "optuna": {
            "n_estimators": {"type": "int", "low": 200, "high": 1000},
            "max_depth": {"type": "int", "low": 3, "high": 15},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0}
        }
    },

    "lgbm": {
        "class": LGBMClassifier,
        "baseline": {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 64,
            "max_depth": -1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "class_weight": "balanced",
            "n_jobs": -1
        },
        "param_grid": {
            "n_estimators": [200, 500, 1000],
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves": [31, 64, 128],
            "max_depth": [-1, 10, 20],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0]
        },
        "optuna": {
            "n_estimators": {"type": "int", "low": 200, "high": 1000},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "num_leaves": {"type": "int", "low": 31, "high": 256},
            "max_depth": {"type": "int", "low": -1, "high": 50},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0}
        }
    }
}


def get_model_config(name: str):
    """
    Retrieve model class and configs.

    Args:
        name (str): One of ['rf', 'xgb', 'lgbm'].

    Returns:
        model_class: Class of the model.
        configs: Dictionary with keys 'baseline', 'param_grid', 'optuna'.
    """
    if name not in MODEL_CONFIGS:
        raise KeyError(f"Model '{name}' not found. Available: {list(MODEL_CONFIGS.keys())}")

    entry = MODEL_CONFIGS[name]
    return entry["class"], {
        "baseline": entry["baseline"],
        "param_grid": entry["param_grid"],
        "optuna": entry["optuna"]
    }