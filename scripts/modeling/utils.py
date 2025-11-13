import joblib
import logging
import os
import matplotlib.pyplot as plt
from tabulate import tabulate

def setup_logger():
    """
    Configure and return a logger instance
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers= [
            logging.StreamHandler(),
            logging.FileHandler("pipeline.log", mode="a")
        ]
    )

    return logging.getLogger(__name__)

# Configure logger
logger = setup_logger()


def format_leaderboard(leaderboard, sort_by: str = "roc_auc"):
    """
    Format the leaderboard by sorting.

    Args:
        leaderboard (List[Dict[str, Any]]): The leaderboard to format.
        sort_by (str): The column to sort by. Defaults to "roc_auc".

    Returns:
        str: The formatted leaderboard as a string.
    """
    
    rows = []
    for entry in leaderboard:
        model = entry["model"]
        metrics = entry["scores"].get("metrics", {})
        rows.append([
            model.upper(),
            metrics.get("roc_auc", 0),
            metrics.get("f1", 0),
            metrics.get("precision", 0),
            metrics.get("recall", 0)
        ])

    # Sort leaderboard
    rows.sort(key=lambda x: x[1], reverse=True)

    header = ["Model", "ROC-AUC", "F1", "Precision", "Recall"]
    table = [header] + rows

    col_widths = [max(len(str(cell)) for cell in col) for col in zip(*table)]
    formatted = [
        "  ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths))
        for row in table
    ]
    return "\n".join(formatted)


def save_model(model, path: str):
    """
    Save a trained model to disk using joblib.
    
    Args:
        model: Trained model object.
        path (str): File path where model should be saved.
    """
    try:
        joblib.dump(model, path)
        logger.info(f"Model saved successfully at {path}")
    except Exception as e:
        logger.error(f"Error saving model at {path}: {e}")
        raise


def load_model(path: str):
    """
    Load a trained model from disk.
    
    Args:
        path (str): File path to the saved model.
    
    Returns:
        Loaded model object.
    """
    try:
        model = joblib.load(path)
        logger.info(f"Model loaded successfully from {path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {path}: {e}")
        raise


def save_plot(save_path: str):
    """
    Save the current matplotlib figure to the specified path.
    Creates directories if they don't exist.
    
    Args:
        save_path (str): File path where the plot should be saved.
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save the current figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved successfully at {save_path}")
    except Exception as e:
        logger.error(f"Error saving plot at {save_path}: {e}")
        raise

