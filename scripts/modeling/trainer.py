import logging
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from .evaluator import Evaluator
from .model_configs import get_model_config
from .utils import setup_logger
import optuna
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Configure logger
logger = setup_logger()


class Trainer:
    """
    A class used to train models with optional hyperparameter tuning.
    """
    def __init__(self, model_name: str, tuning: str = "random", n_iter: int = 20):
        """
        Initialize the Trainer with various parameters.
        Args:
            model_name (str): Model key ("rf", "xgb", "lgbm").
            tuning (str, optional): The method to use for hyperparameter tuning; can be "none", "random", "optuna", or "grid". Defaults to "random".
            n_iter (int, optional): The number of iterations for search. Defaults to 20.
        """

        self.model_name = model_name
        self.tuning = tuning
        self.n_iter = n_iter
        self.evaluator = Evaluator()
        self.best_model = None

    def train(self, X_train, y_train, X_val, y_val):
        """
        Train a model with optional hyperparameter tuning.

        Args:
            X_train (pd.DataFrame): The training features.
            y_train (pd.Series): The target variable for training.
            X_val (pd.DataFrame): The validation features.
            y_val (pd.Series): The target variable for validation.

        Returns:
            Dict[str, Any]: A dictionary containing the evaluation results and the best model.
        """

        model_class, configs = get_model_config(self.model_name)
        model = model_class(**configs["baseline"])

        if self.tuning == "none":
            logger.info(f"Training {self.model_name} with baseline params...")
            model.fit(X_train, y_train)
            result = self.evaluator.evaluate(model, X_val, y_val, self.model_name, model.get_params())
            self.best_model = model

        elif self.tuning == "random":
            logger.info(f"Running Random Search for {self.model_name}...")
            param_distributions = self._param_space()
            search = RandomizedSearchCV(model, param_distributions, n_iter=self.n_iter,
                                        scoring="f1_weighted", cv=3, n_jobs=-1, verbose=1)
            search.fit(X_train, y_train)
            self.best_model = search.best_estimator_
            result = self.evaluator.evaluate(self.best_model, X_val, y_val,
                                             self.model_name, search.best_params_)

        elif self.tuning == "grid":
            logger.info(f"Running Grid Search for {self.model_name}...")
            param_grid = self._param_space(grid=True)
            search = GridSearchCV(model, param_grid, scoring="f1_weighted",
                                  cv=3, n_jobs=-1, verbose=1)
            search.fit(X_train, y_train)
            self.best_model = search.best_estimator_
            result = self.evaluator.evaluate(self.best_model, X_val, y_val,
                                             self.model_name, search.best_params_)

        elif self.tuning == "optuna":
            logger.info(f"Running Optuna optimization for {self.model_name}...")
            def objective(trial):
                params = self._sample_params(trial)
                model_class, configs = get_model_config(self.model_name)
                model = model_class(**configs["baseline"])
                model.set_params(**params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                return f1_score(y_val, y_pred, average="weighted")

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=self.n_iter)
            best_params = study.best_params
            model_class, configs = get_model_config(self.model_name)
            self.best_model = model_class(**configs["baseline"])
            self.best_model.set_params(**best_params)
            self.best_model.fit(X_train, y_train)
            result = self.evaluator.evaluate(self.best_model, X_val, y_val,
                                             self.model_name, best_params)

        else:
            raise ValueError(f"Unknown tuning method: {self.tuning}")

        # Save best model
        self.evaluator.save_best_model(self.best_model, self.model_name)
        return result

    def _param_space(self, grid: bool = False):
        """
        Define search space for Random/Grid search.

        Args:
            grid (bool, optional): Whether to use a grid or randomized search. Defaults to False.
        """

        if self.model_name == "rf":
            return {
                "n_estimators": [100, 200, 500] if grid else np.arange(100, 1000, 100),
                "max_depth": [None, 10, 20, 30] if grid else np.arange(5, 50, 5),
                "min_samples_split": [2, 5, 10],
            }

        elif self.model_name == "xgb":
            return {
                "n_estimators": [100, 200, 500],
                "max_depth": [3, 6, 10],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.6, 0.8, 1.0],
            }

        elif self.model_name == "lgbm":
            return {
                "n_estimators": [100, 200, 500],
                "learning_rate": [0.01, 0.05, 0.1],
                "num_leaves": [31, 64, 128],
                "subsample": [0.6, 0.8, 1.0],
            }

        return {}

    def _sample_params(self, trial):
        """
        Define search space for Optuna.

        Args:
            trial (optuna.Trial): The current trial.
        """
        
        if self.model_name == "rf":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
                "max_depth": trial.suggest_int("max_depth", 5, 50),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            }

        elif self.model_name == "xgb":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            }

        elif self.model_name == "lgbm":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                "num_leaves": trial.suggest_int("num_leaves", 31, 256),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            }

        return {}
