import os
import json
import joblib
from .utils import setup_logger
from typing import Dict, Any, List
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Configure logger
logger = setup_logger()

class Evaluator:
    """
    A class used to evaluate a trained model on validation data and track its performance.
    """

    def __init__(self, leaderboard_path: str = "models/leaderboard.json", model_dir: str = "models/"):
        """
        Initialize the Evaluator with various parameters.

        Args:
            leaderboard_path (str, optional): The path to save the leaderboard. Defaults to "models/leaderboard.json".
            model_dir (str, optional): The directory where trained models will be saved. Defaults to "models/".
        """

        self.leaderboard_path = leaderboard_path
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.leaderboard: List[Dict[str, Any]] = []


    def evaluate(self, model, X_val, y_val, model_name: str, params: Dict[str, Any]):
        """
        Evaluate a trained model on validation data, log metrics,
        and update leaderboard.

        Args:
            model: The trained model.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation target variable.
            model_name (str): The name of the model being evaluated.
            params (Dict[str, Any]): Model hyperparameters.
        """
        y_pred = model.predict(X_val)
        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_val)[:, 1]
            except Exception:
                pass

        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_val, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_val, y_pred, average="weighted", zero_division=0),
        }
        if y_proba is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_val, y_proba)
            except Exception:
                pass

        result = {
            "model": model_name,
            "params": params,
            "metrics": metrics,
        }

        # Log metrics
        logger.info(f"Evaluation for {model_name} | Metrics: {metrics}")

        # Update leaderboard
        self.leaderboard.append(result)
        self._save_leaderboard()

        return result

    def get_best(self, metric: str = "f1") -> Dict[str, Any]:
        """
        Get best model result from leaderboard based on given metric.

        Args:
            metric (str, optional): The metric to use for comparison. Defaults to "f1".

        Returns:
            Dict[str, Any]: The best model result.
        """

        if not self.leaderboard:
            raise ValueError("Leaderboard is empty. Run evaluations first.")
        best = max(self.leaderboard, key=lambda x: x["metrics"].get(metric, 0))
        logger.info(f"Best model so far ({metric}): {best['model']} | {best['metrics'][metric]:.4f}")
        return best

    def save_best_model(self, model, model_name: str):
        """
        Save trained model artifact and its params.

        Args:
            model (object): The trained model.
            model_name (str): The name of the model being saved.
        """
        best_model_path = os.path.join(self.model_dir, f"{model_name}_best.pkl")
        joblib.dump(model, best_model_path)
        logger.info(f"Saved best model: {best_model_path}")

    def _save_leaderboard(self):
        with open(self.leaderboard_path, "w") as f:
            json.dump(self.leaderboard, f, indent=4)