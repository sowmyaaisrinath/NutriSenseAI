"""
Model evaluation and metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from sklearn.calibration import calibration_curve
from typing import Dict, List, Any, Optional
import warnings

warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Comprehensive model evaluation and visualization
    """
    
    def __init__(self, task_type: str = 'multiclass'):
        """
        Initialize evaluator
        
        Args:
            task_type: 'binary' or 'multiclass'
        """
        self.task_type = task_type
        self.results = {}
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, 
                      y_test: pd.Series, model_name: str = 'Model') -> Dict:
        """
        Comprehensive evaluation of a single model
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: True labels
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}")
        print('='*60)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None
        
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Print results
        self._print_metrics(metrics)
        
        # Store results
        self.results[model_name] = {
            'metrics': metrics,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test
        }
        
        return metrics
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray,
                          y_pred_proba: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate all evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # For multiclass, use macro averaging
        average = 'binary' if self.task_type == 'binary' else 'macro'
        
        metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Classification report
        metrics['classification_report'] = classification_report(y_true, y_pred, zero_division=0)
        
        # AUC metrics (if probabilities available)
        if y_pred_proba is not None:
            try:
                if self.task_type == 'binary':
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                    metrics['avg_precision'] = average_precision_score(y_true, y_pred_proba[:, 1])
                else:  # multiclass
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, 
                                                       multi_class='ovr', average='macro')
            except Exception as e:
                print(f"Warning: Could not calculate AUC metrics: {e}")
        
        return metrics
    
    def _print_metrics(self, metrics: Dict) -> None:
        """
        Print evaluation metrics
        
        Args:
            metrics: Dictionary with metrics
        """
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        
        if 'roc_auc' in metrics:
            print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
        
        print(f"\nClassification Report:")
        print(metrics['classification_report'])
        
        print(f"\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
    
    def compare_models(self, models_dict: Dict[str, Any], 
                      X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            models_dict: Dictionary mapping model names to model objects
            X_test: Test features
            y_test: True labels
            
        Returns:
            DataFrame with comparison results
        """
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        comparison_results = []
        
        for name, model in models_dict.items():
            metrics = self.evaluate_model(model, X_test, y_test, name)
            
            result = {
                'Model': name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1']
            }
            
            if 'roc_auc' in metrics:
                result['ROC-AUC'] = metrics['roc_auc']
            
            comparison_results.append(result)
        
        # Create comparison DataFrame
        df_comparison = pd.DataFrame(comparison_results)
        df_comparison = df_comparison.sort_values('F1-Score', ascending=False)
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(df_comparison.to_string(index=False))
        
        return df_comparison
    
    def plot_confusion_matrix(self, model_name: str, 
                            class_names: Optional[List[str]] = None,
                            figsize: tuple = (8, 6)) -> None:
        """
        Plot confusion matrix for a model
        
        Args:
            model_name: Name of the model
            class_names: Optional class names for labels
            figsize: Figure size
        """
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not found in results")
        
        cm = self.results[model_name]['metrics']['confusion_matrix']
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_feature_importance(self, model: Any, feature_names: List[str],
                               top_n: int = 20, figsize: tuple = (10, 8)) -> None:
        """
        Plot feature importance for tree-based models
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            top_n: Number of top features to display
            figsize: Figure size
        """
        if not hasattr(model, 'feature_importances_'):
            print(f"Model does not have feature_importances_ attribute")
            return
        
        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        # Create DataFrame
        df_importance = pd.DataFrame({
            'Feature': [feature_names[i] for i in indices],
            'Importance': importances[indices]
        })
        
        # Plot
        plt.figure(figsize=figsize)
        sns.barplot(data=df_importance, x='Importance', y='Feature')
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_roc_curves(self, figsize: tuple = (10, 8)) -> None:
        """
        Plot ROC curves for all evaluated models (binary classification)
        
        Args:
            figsize: Figure size
        """
        if self.task_type != 'binary':
            print("ROC curves only available for binary classification")
            return
        
        plt.figure(figsize=figsize)
        
        for model_name, result in self.results.items():
            if result['y_pred_proba'] is not None:
                y_test = result['y_test']
                y_pred_proba = result['y_pred_proba'][:, 1]
                
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                auc = roc_auc_score(y_test, y_pred_proba)
                
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_calibration_curve(self, model_name: str, n_bins: int = 10,
                              figsize: tuple = (8, 8)) -> None:
        """
        Plot calibration curve to assess probability calibration
        
        Args:
            model_name: Name of the model
            n_bins: Number of bins for calibration curve
            figsize: Figure size
        """
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not found in results")
        
        result = self.results[model_name]
        
        if result['y_pred_proba'] is None:
            print(f"Model {model_name} does not have probability predictions")
            return
        
        y_test = result['y_test']
        
        if self.task_type == 'binary':
            y_pred_proba = result['y_pred_proba'][:, 1]
        else:
            print("Calibration curves only shown for binary classification")
            return
        
        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=n_bins)
        
        # Plot
        plt.figure(figsize=figsize)
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        plt.plot(prob_pred, prob_true, marker='o', label=model_name)
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title(f'Calibration Curve - {model_name}')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, 
                              n_classes=3, n_informative=15,
                              random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train a model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    evaluator = ModelEvaluator(task_type='multiclass')
    metrics = evaluator.evaluate_model(model, X_test, y_test, 'Random Forest')

