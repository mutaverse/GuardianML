import os
import logging
import shap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from .utils import save_plot, setup_logger

# Configure logger
logger = setup_logger()



class Explainer:
    """
    A class used to explain the behavior of the models.

    Attributes:
        model: The trained machine learning model.
        X_train: The training features.
        X_test: The testing features.
        y_test: The target variable for testing (optional).
        feature_names: The names of the features in the dataset.
        class_names: The names of the classes in the dataset (optional).
    """
    def __init__(self, model, X_train, X_test, feature_names, y_test = None, class_names=None):
        """
        Initialize the Explainer with various parameters.
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.class_names = class_names

        # Initialize SHAP explainer
        self.shap_explainer = shap.TreeExplainer(model) if hasattr(model, "predict_proba") else shap.Explainer(model)

    # -------------------
    # SHAP Summary Plot
    # -------------------
    def shap_summary_plot(self, shap_values, X_sample, save_path, max_display=20):
        """
        Generate a summary plot of the SHAP values for the given features.

        Args:
            shap_values (shap.TreeExplainer): The SHAP explainer.
            X_sample (pd.DataFrame): The sample features to evaluate.
            save_path (str): The path to save the plot.
            max_display (int, optional): The maximum number of features to display. Defaults to 20.

        Returns:
            str: The path to the saved plot.
        """

        plt.figure()
        shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, show=False, max_display=max_display)
        save_plot(save_path)
        plt.close()
        return save_path

    # -------------------
    # SHAP Feature Importance (Top N)
    # -------------------
    def shap_feature_importance(self, shap_values, save_path, top_n=20):
        """
        Generate a plot of the feature importance for the given features.

        Args:
            shap_values (shap.TreeExplainer): The SHAP explainer.
            save_path (str): The path to save the plot.
            top_n (int, optional): The number of features to display. Defaults to 20.

        Returns:
            str: The path to the saved plot.
        """

        importance = np.abs(shap_values.values).mean(axis=0)
        importance_df = (
            pd.DataFrame({"feature": self.feature_names, "importance": importance})
            .sort_values(by="importance", ascending=False)
            .head(top_n)
        )

        plt.figure(figsize=(10, 6))
        sns.barplot(
            x="importance",
            y="feature",
            data=importance_df,
            palette="coolwarm"
        )
        plt.title(f"Top {top_n} Features by Mean SHAP Importance", fontsize=13, fontweight="bold")
        plt.xlabel("Mean |SHAP Value|")
        plt.ylabel("Feature")
        plt.tight_layout()
        save_plot(save_path)
        plt.close()
        return save_path

    # -------------------
    # Confusion Matrix 
    # -------------------
    def plot_confusion_matrix(self, model, X_test, y_test, save_path):
        """
        Generate confusion matrix to explain how the model predicts each class.

        Args:
            model: The trained machine learning model.
            X_test (pd.DataFrame): The testing features.
            y_test (pd.Series): The target variable for testing.
            save_path (str): The path to save the plot.
        """
        y_pred = model.predict(X_test)
        disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap = "Blues")
        disp.ax_.set_title("Confusion Matrix")
        disp.figure_.savefig(save_path, bbox_inches = "tight")
        plt.close()

    
    # -------------------
    # Generate Explainability Report
    # -------------------
    def generate_all_reports(self, report_dir="reports/plots", sample_size=500):
        """
        Generate an explainability report using SHAP values.

        Args:
            report_dir (str, optional): The directory to save the reports. Defaults to "reports/plots".
            sample_size (int, optional): The number of samples to use for computing SHAP values. Defaults to 500.
        """

        os.makedirs(report_dir, exist_ok=True)

        # Use subset for performance
        X_sample = self.X_test
        if len(self.X_test) > sample_size:
            X_sample = self.X_test.sample(sample_size, random_state=42)

        logger.info("[Explainer] Computing SHAP values (this may take a few seconds)...")
        shap_values = self.shap_explainer(X_sample)

        logger.info("[Explainer] Generating SHAP plots...")
        summary_path = os.path.join(report_dir, "shap_summary.png")
        importance_path = os.path.join(report_dir, "feature_importance.png")

        self.shap_summary_plot(shap_values, X_sample, summary_path, max_display=20)
        self.shap_feature_importance(shap_values, importance_path, top_n=20)

        conf_path = os.path.join(report_dir, "confusion_matrix.png")
        self.plot_confusion_matrix(self.model, self.X_test, self.y_test, conf_path)

        # HTML summary page
        index_path = os.path.join(report_dir, "index.html")
        with open(index_path, "w") as f:
            f.write(f"""
            <html>
            <head><title>Model Explainability Report</title></head>
            <body style="font-family: Arial; margin: 40px;">
                <h1>Model Explainability Report</h1>
                <p>Generated using SHAP (Top 20 Features)</p>
        
                <h2>Feature Importance</h2>
                <img src="feature_importance.png" width="800"><br>
                
                <h2>Confusion Matrix</h2>
                <img src="confusion_matrix.png" width="800"><br>
        
                <h2>SHAP Summary</h2>
                <img src="shap_summary.png" width="800"><br>
            </body>
            </html>
            """)
        
        print(f"[Explainer] Explainability report saved to {report_dir}")