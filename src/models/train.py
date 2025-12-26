"""
Model training pipeline for nutrition classification
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from typing import Dict, Tuple, Any, List
import joblib
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class NutritionClassifier:
    """
    Train and manage nutrition classification models
    """
    
    def __init__(self, task_type: str = 'multiclass'):
        """
        Initialize classifier
        
        Args:
            task_type: 'binary' for healthy/unhealthy or 'multiclass' for low/medium/high
        """
        self.task_type = task_type
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        
    def get_models(self) -> Dict[str, Any]:
        """
        Get dictionary of models to train
        
        Returns:
            Dictionary mapping model names to model objects
        """
        if self.task_type == 'binary':
            models = {
                'Logistic Regression': LogisticRegression(
                    max_iter=1000, 
                    random_state=42,
                    class_weight='balanced'
                ),
                'Decision Tree': DecisionTreeClassifier(
                    random_state=42,
                    class_weight='balanced'
                ),
                'Random Forest': RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    class_weight='balanced',
                    n_jobs=-1
                ),
                'XGBoost': XGBClassifier(
                    n_estimators=100,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='logloss'
                ),
                'MLP': MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    max_iter=500,
                    random_state=42,
                    early_stopping=True
                )
            }
        else:  # multiclass
            models = {
                'Logistic Regression': LogisticRegression(
                    max_iter=1000,
                    multi_class='multinomial',
                    random_state=42,
                    class_weight='balanced'
                ),
                'Decision Tree': DecisionTreeClassifier(
                    random_state=42,
                    class_weight='balanced'
                ),
                'Random Forest': RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    class_weight='balanced',
                    n_jobs=-1
                ),
                'XGBoost': XGBClassifier(
                    n_estimators=100,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='mlogloss',
                    objective='multi:softmax'
                ),
                'MLP': MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    max_iter=500,
                    random_state=42,
                    early_stopping=True
                )
            }
        
        return models
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict:
        """
        Train all models and track performance
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary with training results
        """
        print(f"\nTraining models for {self.task_type} classification...")
        print(f"Training set size: {len(X_train)}")
        print(f"Number of features: {X_train.shape[1]}")
        print(f"Label distribution:\n{y_train.value_counts()}\n")
        
        models = self.get_models()
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Store model
                self.models[name] = model
                
                # Evaluate on training set
                train_score = model.score(X_train, y_train)
                
                # Evaluate on validation set if provided
                if X_val is not None and y_val is not None:
                    val_score = model.score(X_val, y_val)
                else:
                    val_score = None
                
                results[name] = {
                    'model': model,
                    'train_score': train_score,
                    'val_score': val_score
                }
                
                print(f"  Train accuracy: {train_score:.4f}")
                if val_score is not None:
                    print(f"  Val accuracy: {val_score:.4f}")
                print()
                
            except Exception as e:
                print(f"  Error training {name}: {e}\n")
                continue
        
        # Identify best model based on validation score (or training if no validation)
        if results:
            score_key = 'val_score' if X_val is not None else 'train_score'
            valid_results = {k: v for k, v in results.items() if v[score_key] is not None}
            
            if valid_results:
                best_name = max(valid_results, key=lambda k: valid_results[k][score_key])
                self.best_model_name = best_name
                self.best_model = self.models[best_name]
                print(f"Best model: {best_name} ({score_key}: {valid_results[best_name][score_key]:.4f})")
        
        return results
    
    def hyperparameter_tuning(self, model_name: str, X_train: pd.DataFrame, 
                            y_train: pd.Series, param_grid: Dict) -> Any:
        """
        Perform hyperparameter tuning for a specific model
        
        Args:
            model_name: Name of model to tune
            X_train: Training features
            y_train: Training labels
            param_grid: Dictionary of parameters to search
            
        Returns:
            Best estimator from GridSearchCV
        """
        print(f"\nHyperparameter tuning for {model_name}...")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Train models first.")
        
        base_model = self.models[model_name]
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Update model with best estimator
        self.models[model_name] = grid_search.best_estimator_
        
        return grid_search.best_estimator_
    
    def save_model(self, model_name: str, filepath: str) -> None:
        """
        Save a trained model to disk
        
        Args:
            model_name: Name of model to save
            filepath: Path to save the model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.models[model_name], filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str, model_name: str = None) -> Any:
        """
        Load a trained model from disk
        
        Args:
            filepath: Path to the saved model
            model_name: Optional name to assign to the loaded model
            
        Returns:
            Loaded model
        """
        model = joblib.load(filepath)
        
        if model_name:
            self.models[model_name] = model
        
        print(f"Model loaded from {filepath}")
        return model


def get_default_param_grids() -> Dict[str, Dict]:
    """
    Get default hyperparameter grids for tuning
    
    Returns:
        Dictionary mapping model names to parameter grids
    """
    param_grids = {
        'Logistic Regression': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        },
        'Decision Tree': {
            'max_depth': [3, 5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0]
        },
        'MLP': {
            'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100)],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
    }
    
    return param_grids


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=15, 
                              n_classes=3, n_informative=10,
                              random_state=42)
    
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train models
    classifier = NutritionClassifier(task_type='multiclass')
    results = classifier.train_models(X_train, y_train, X_test, y_test)
    
    print(f"\nBest model: {classifier.best_model_name}")

