import os
from typing import List, Optional
import numpy as np
from .model_configs import MODEL_CONFIGS
from .trainer import Trainer
from .evaluator import Evaluator
from .utils import format_leaderboard, save_model, setup_logger
from .explainer import Explainer

# Configure logger
logger = setup_logger()


class Orchestrator:
    """
    A class used to orchestrate the training of multiple models and evaluate their performance.
    """

    def __init__(self, models: Optional[List[str]] = None, tuning: str = "random",
                 output_dir: str = "models", report_dir: str = "reports/plots"):
        """
        Initialize the Orchestrator with various parameters.

        Args:
            models (List[str], optional): A list of model names to use. Defaults to a list of all model configurations.
            tuning (str, optional): The method to use for hyperparameter tuning; can be "none", "random", "optuna", or "grid". Defaults to "random".
            output_dir (str, optional): The directory to save the models. Defaults to "models/".
            report_dir (str, optional): The directory to save the explainability reports. Defaults to "reports/plots/".
        """

        self.models = models if models is not None else list(MODEL_CONFIGS.keys())
        self.tuning = tuning
        self.output_dir = output_dir
        self.report_dir = report_dir
        self.leaderboard = []

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(report_dir, exist_ok=True)

    def run(self, X_train, y_train, X_val, y_val, class_weights=None):
        """
        Run the orchestration of multiple models.

        Args:
            X_train (pd.DataFrame): The training features.
            y_train (pd.Series): The target variable for training.
            X_val (pd.DataFrame): The validation features.
            y_val (pd.Series): The target variable for validation.
            class_weights (Optional[List[float]]): Class weights to use.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the evaluation results and model information.
        """
        
        logger.info(f"Starting orchestration with models: {self.models}")

        best_model = None
        best_score = -1
        best_model_name = None
        best_params_used = None

        for model_name in self.models:
            logger.info(f"--- Training {model_name.upper()} ---")

            trainer = Trainer(model_name=model_name, tuning=self.tuning)
            result = trainer.train(X_train, y_train, X_val, y_val)

            trained_model = trainer.best_model
            best_params = result["params"]

            evaluator = Evaluator()
            scores = evaluator.evaluate(
                model=trained_model,
                X_val=X_val,
                y_val=y_val,
                model_name=model_name,
                params=best_params
            )

            result = {"model": model_name, "params": best_params, "scores": scores}
            self.leaderboard.append(result)
            logger.info(f"Scores for {model_name.upper()}: {scores}")

            current_score = scores["metrics"].get("roc_auc", 0)
            if current_score > best_score:
                best_score = current_score
                best_model = trained_model
                best_model_name = model_name
                best_params_used = best_params

        # Save and explain the best model
        if best_model is not None:
            save_path = os.path.join(self.output_dir, f"best_{best_model_name}.pkl")
            save_model(best_model, save_path)

            logger.info(f"Best model: {best_model_name.upper()} | ROC-AUC={best_score:.4f}")
            logger.info(f"Saved best model to {save_path}")
            logger.info(f"Best params: {best_params_used}")

            # Explain best model (using small sample for speed)
            logger.info(f"Generating SHAP explainability report for {best_model_name.upper()} (Top 20 features)...")
            explainer = Explainer(
                model=best_model,
                X_train=X_train,
                X_test=X_val,
                y_test = y_val,
                feature_names=X_train.columns.tolist(),
                class_names=list(np.unique(y_train)) if y_train is not None else None
            )
            explainer.generate_all_reports(report_dir=self.report_dir, sample_size=500)
            logger.info(f"Explainability report saved under {self.report_dir}")

        logger.info("\n" + format_leaderboard(self.leaderboard))
        return self.leaderboard