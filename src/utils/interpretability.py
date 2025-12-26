"""
Model interpretability using SHAP and other techniques
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from typing import Any, List, Optional
import warnings

warnings.filterwarnings('ignore')


class ModelInterpreter:
    """
    Interpret model predictions using SHAP and feature importance
    """
    
    def __init__(self, model: Any, X_train: pd.DataFrame):
        """
        Initialize interpreter
        
        Args:
            model: Trained model
            X_train: Training data (used for SHAP background)
        """
        self.model = model
        self.X_train = X_train
        self.explainer = None
        self.shap_values = None
    
    def explain_with_shap(self, X_test: pd.DataFrame, 
                         sample_size: Optional[int] = 100) -> None:
        """
        Generate SHAP explanations for the model
        
        Args:
            X_test: Test data to explain
            sample_size: Number of background samples to use (for speed)
        """
        print("Generating SHAP explanations...")
        
        # Sample background data for faster computation
        if sample_size and len(self.X_train) > sample_size:
            background = shap.sample(self.X_train, sample_size)
        else:
            background = self.X_train
        
        # Create explainer based on model type
        try:
            # Try TreeExplainer for tree-based models (faster)
            self.explainer = shap.TreeExplainer(self.model)
            self.shap_values = self.explainer.shap_values(X_test)
        except:
            try:
                # Fallback to KernelExplainer for other models
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba, 
                    background
                )
                self.shap_values = self.explainer.shap_values(X_test)
            except Exception as e:
                print(f"Error creating SHAP explainer: {e}")
                return
        
        print("SHAP explanations generated successfully")
    
    def plot_summary(self, X_test: pd.DataFrame, 
                    class_idx: Optional[int] = None,
                    max_display: int = 20) -> None:
        """
        Plot SHAP summary plot
        
        Args:
            X_test: Test data
            class_idx: Class index for multiclass (None for binary)
            max_display: Maximum number of features to display
        """
        if self.shap_values is None:
            print("Run explain_with_shap() first")
            return
        
        print("Plotting SHAP summary...")
        
        # Handle multiclass vs binary
        if isinstance(self.shap_values, list) and class_idx is not None:
            shap_vals = self.shap_values[class_idx]
        elif isinstance(self.shap_values, list):
            shap_vals = self.shap_values[1]  # Positive class for binary
        else:
            shap_vals = self.shap_values
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_vals, X_test, max_display=max_display, show=False)
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, X_test: pd.DataFrame,
                               class_idx: Optional[int] = None,
                               max_display: int = 20) -> None:
        """
        Plot SHAP feature importance (bar plot)
        
        Args:
            X_test: Test data
            class_idx: Class index for multiclass
            max_display: Maximum number of features to display
        """
        if self.shap_values is None:
            print("Run explain_with_shap() first")
            return
        
        print("Plotting SHAP feature importance...")
        
        # Handle multiclass vs binary
        if isinstance(self.shap_values, list) and class_idx is not None:
            shap_vals = self.shap_values[class_idx]
        elif isinstance(self.shap_values, list):
            shap_vals = self.shap_values[1]
        else:
            shap_vals = self.shap_values
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_vals, X_test, plot_type="bar", 
                         max_display=max_display, show=False)
        plt.tight_layout()
        plt.show()
    
    def explain_single_prediction(self, instance: pd.DataFrame, 
                                 class_idx: Optional[int] = None) -> None:
        """
        Explain a single prediction using waterfall plot
        
        Args:
            instance: Single instance to explain (as DataFrame with one row)
            class_idx: Class index for multiclass
        """
        if self.explainer is None:
            print("Run explain_with_shap() first")
            return
        
        print("Explaining single prediction...")
        
        # Get SHAP values for this instance
        if hasattr(self.explainer, 'shap_values'):
            shap_vals = self.explainer.shap_values(instance)
        else:
            shap_vals = self.shap_values[0]  # Use first from batch
        
        # Handle multiclass
        if isinstance(shap_vals, list):
            if class_idx is not None:
                shap_vals = shap_vals[class_idx]
            else:
                shap_vals = shap_vals[1]  # Positive class
        
        # Create explanation object for waterfall plot
        if isinstance(shap_vals, np.ndarray):
            shap_vals = shap_vals[0] if len(shap_vals.shape) > 1 else shap_vals
        
        # Waterfall plot
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_vals,
                base_values=self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
                data=instance.values[0],
                feature_names=instance.columns.tolist()
            ),
            show=False
        )
        plt.tight_layout()
        plt.show()
    
    def plot_dependence(self, feature: str, X_test: pd.DataFrame,
                       interaction_feature: Optional[str] = None,
                       class_idx: Optional[int] = None) -> None:
        """
        Plot SHAP dependence plot showing how a feature affects predictions
        
        Args:
            feature: Feature to analyze
            X_test: Test data
            interaction_feature: Optional feature for interaction coloring
            class_idx: Class index for multiclass
        """
        if self.shap_values is None:
            print("Run explain_with_shap() first")
            return
        
        print(f"Plotting dependence for {feature}...")
        
        # Handle multiclass
        if isinstance(self.shap_values, list):
            if class_idx is not None:
                shap_vals = self.shap_values[class_idx]
            else:
                shap_vals = self.shap_values[1]
        else:
            shap_vals = self.shap_values
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature, 
            shap_vals, 
            X_test,
            interaction_index=interaction_feature,
            show=False
        )
        plt.tight_layout()
        plt.show()


def get_top_features_by_importance(model: Any, feature_names: List[str], 
                                  top_n: int = 10) -> pd.DataFrame:
    """
    Get top N most important features from a model
    
    Args:
        model: Trained model with feature_importances_
        feature_names: List of feature names
        top_n: Number of top features to return
        
    Returns:
        DataFrame with feature names and importances
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute")
        return pd.DataFrame()
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    df = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices],
        'Importance': importances[indices]
    })
    
    return df


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=20, 
                              n_informative=15, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Interpret
    interpreter = ModelInterpreter(model, X_train)
    interpreter.explain_with_shap(X_test.head(50))
    interpreter.plot_summary(X_test.head(50))

