"""
Prediction Demo - Using Trained Models
=======================================
This script demonstrates how to use trained models to make predictions on new data.
"""

import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Add project root to path
sys.path.append('.')

from src.data.preprocessing import NutritionPreprocessor


class NutritionPredictor:
    """Class for making predictions with trained models."""
    
    def __init__(self, task_name: str, models_dir: str = 'models'):
        """
        Initialize predictor for a specific task.
        
        Parameters:
        -----------
        task_name : str
            Name of the task (e.g., 'sugar_class', 'healthy')
        models_dir : str
            Directory containing saved models
        """
        self.task_name = task_name
        self.models_dir = Path(models_dir)
        
        # Load model artifacts
        print(f"Loading model for task: {task_name}")
        
        model_path = self.models_dir / f'best_model_{task_name}.pkl'
        scaler_path = self.models_dir / f'scaler_{task_name}.pkl'
        features_path = self.models_dir / f'features_{task_name}.pkl'
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Please train the model first using train_all_tasks.py"
            )
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = joblib.load(features_path)
        
        print(f"✓ Loaded model: {model_path.name}")
        print(f"✓ Features: {len(self.feature_names)}")
        
        # Define label mappings
        self.label_mappings = {
            'healthy': {0: 'Unhealthy', 1: 'Healthy'},
            'sugar_class': {0: 'Low Sugar', 1: 'Medium Sugar', 2: 'High Sugar'},
            'fiber_class': {0: 'Low Fiber', 1: 'Medium Fiber', 2: 'High Fiber'},
            'protein_class': {0: 'Low Protein', 1: 'Medium Protein', 2: 'High Protein'},
            'sodium_class': {0: 'Low Sodium', 1: 'Medium Sodium', 2: 'High Sodium'}
        }
    
    def preprocess_input(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess input data for prediction.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data with nutrition information
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed features ready for prediction
        """
        # Engineer features if not already present
        preprocessor = NutritionPreprocessor()
        
        # Check if engineered features exist
        engineered_features = ['energy_density', 'fat_ratio', 'carb_ratio', 'protein_ratio']
        if not all(f in data.columns for f in engineered_features):
            print("Engineering features...")
            data = preprocessor.engineer_features(data)
        
        # Select only the features used during training
        missing_features = [f for f in self.feature_names if f not in data.columns]
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            # Add missing features with zeros
            for feature in missing_features:
                data[feature] = 0
        
        X = data[self.feature_names].copy()
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=self.feature_names,
            index=X.index
        )
        
        return X_scaled
    
    def predict(self, data: pd.DataFrame, return_proba: bool = False) -> pd.DataFrame:
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data with nutrition information
        return_proba : bool
            If True, return prediction probabilities
            
        Returns:
        --------
        pd.DataFrame
            Predictions with labels and optionally probabilities
        """
        X_scaled = self.preprocess_input(data)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Convert to labels
        label_map = self.label_mappings.get(self.task_name, {})
        predicted_labels = [label_map.get(p, f"Class {p}") for p in predictions]
        
        results = pd.DataFrame({
            'prediction_numeric': predictions,
            'prediction_label': predicted_labels
        }, index=data.index)
        
        # Add probabilities if requested
        if return_proba and hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_scaled)
            
            # Add probability columns
            for i, label in label_map.items():
                if i < probabilities.shape[1]:
                    results[f'prob_{label}'] = probabilities[:, i]
        
        return results
    
    def predict_single(self, nutrition_info: dict) -> dict:
        """
        Make prediction for a single food item.
        
        Parameters:
        -----------
        nutrition_info : dict
            Dictionary with nutrition information (per 100g)
            Example: {
                'energy_100g': 500,
                'fat_100g': 20,
                'carbohydrates_100g': 60,
                'proteins_100g': 10,
                ...
            }
            
        Returns:
        --------
        dict
            Prediction result with label and probabilities
        """
        # Convert to DataFrame
        df = pd.DataFrame([nutrition_info])
        
        # Make prediction
        result = self.predict(df, return_proba=True)
        
        return result.iloc[0].to_dict()


def demo_predictions():
    """Demonstrate predictions with example data."""
    print("="*80)
    print("NUTRITION PREDICTION DEMO")
    print("="*80)
    
    # Example 1: Predict sugar class for different foods
    print("\n📊 Example 1: Predicting Sugar Content Classification")
    print("-"*80)
    
    example_foods = pd.DataFrame([
        {
            'name': 'Chocolate Bar',
            'energy_100g': 540,
            'fat_100g': 31,
            'saturated-fat_100g': 19,
            'carbohydrates_100g': 57,
            'sugars_100g': 54,  # High sugar
            'fiber_100g': 3,
            'proteins_100g': 5,
            'salt_100g': 0.3,
            'sodium_100g': 0.12
        },
        {
            'name': 'Broccoli',
            'energy_100g': 34,
            'fat_100g': 0.4,
            'saturated-fat_100g': 0.1,
            'carbohydrates_100g': 7,
            'sugars_100g': 1.7,  # Low sugar
            'fiber_100g': 2.6,
            'proteins_100g': 2.8,
            'salt_100g': 0.03,
            'sodium_100g': 0.012
        },
        {
            'name': 'Yogurt',
            'energy_100g': 59,
            'fat_100g': 0.4,
            'saturated-fat_100g': 0.3,
            'carbohydrates_100g': 3.6,
            'sugars_100g': 4.7,  # Medium sugar
            'fiber_100g': 0,
            'proteins_100g': 10,
            'salt_100g': 0.05,
            'sodium_100g': 0.02
        }
    ])
    
    try:
        predictor = NutritionPredictor('sugar_class')
        
        # Make predictions
        predictions = predictor.predict(example_foods, return_proba=True)
        
        # Display results
        results_df = pd.concat([
            example_foods[['name', 'sugars_100g']],
            predictions
        ], axis=1)
        
        print("\nPrediction Results:")
        print(results_df.to_string(index=False))
        
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("Please run train_all_tasks.py first to train the models.")
        return
    
    # Example 2: Predict healthy/unhealthy
    print("\n\n📊 Example 2: Predicting Healthy vs Unhealthy")
    print("-"*80)
    
    try:
        predictor = NutritionPredictor('healthy')
        
        predictions = predictor.predict(example_foods, return_proba=True)
        
        results_df = pd.concat([
            example_foods[['name']],
            predictions
        ], axis=1)
        
        print("\nPrediction Results:")
        print(results_df.to_string(index=False))
        
    except FileNotFoundError:
        print("Model for 'healthy' task not found. Skipping this example.")
    
    # Example 3: Single food prediction with custom input
    print("\n\n📊 Example 3: Custom Food Prediction")
    print("-"*80)
    
    custom_food = {
        'energy_100g': 250,
        'fat_100g': 5,
        'saturated-fat_100g': 1,
        'carbohydrates_100g': 45,
        'sugars_100g': 8,
        'fiber_100g': 5,
        'proteins_100g': 8,
        'salt_100g': 0.5,
        'sodium_100g': 0.2
    }
    
    print("Custom food nutrition (per 100g):")
    for key, value in custom_food.items():
        print(f"  {key}: {value}")
    
    try:
        predictor = NutritionPredictor('sugar_class')
        result = predictor.predict_single(custom_food)
        
        print("\nPrediction:")
        print(f"  Class: {result['prediction_label']}")
        print(f"  Probabilities:")
        for key, value in result.items():
            if key.startswith('prob_'):
                print(f"    {key.replace('prob_', '')}: {value:.2%}")
    
    except FileNotFoundError:
        print("Model not found. Skipping this example.")
    
    print("\n" + "="*80)
    print("✅ Demo complete!")
    print("="*80)
    print("\n💡 To use these models in your own code:")
    print("```python")
    print("from predict_new_data import NutritionPredictor")
    print("")
    print("# Initialize predictor")
    print("predictor = NutritionPredictor('sugar_class')")
    print("")
    print("# Make predictions")
    print("predictions = predictor.predict(your_dataframe)")
    print("```")


def interactive_prediction():
    """Interactive mode for making predictions."""
    print("\n" + "="*80)
    print("INTERACTIVE PREDICTION MODE")
    print("="*80)
    
    # List available models
    models_dir = Path('models')
    available_models = []
    
    for task in ['healthy', 'sugar_class', 'fiber_class', 'protein_class', 'sodium_class']:
        model_path = models_dir / f'best_model_{task}.pkl'
        if model_path.exists():
            available_models.append(task)
    
    if not available_models:
        print("\n❌ No trained models found!")
        print("Please run train_all_tasks.py first.")
        return
    
    print(f"\nAvailable models: {', '.join(available_models)}")
    print("\nEnter 'q' to quit")
    
    while True:
        task = input("\nEnter task name: ").strip()
        
        if task.lower() == 'q':
            break
        
        if task not in available_models:
            print(f"Invalid task. Choose from: {', '.join(available_models)}")
            continue
        
        try:
            predictor = NutritionPredictor(task)
            
            print("\nEnter nutrition values (per 100g):")
            nutrition = {}
            
            required_nutrients = [
                'energy_100g', 'fat_100g', 'saturated-fat_100g',
                'carbohydrates_100g', 'sugars_100g', 'fiber_100g',
                'proteins_100g', 'salt_100g', 'sodium_100g'
            ]
            
            for nutrient in required_nutrients:
                value = input(f"  {nutrient}: ")
                try:
                    nutrition[nutrient] = float(value)
                except ValueError:
                    print(f"Invalid value for {nutrient}, using 0")
                    nutrition[nutrient] = 0
            
            # Make prediction
            result = predictor.predict_single(nutrition)
            
            print(f"\n🎯 Prediction: {result['prediction_label']}")
            
            if any(k.startswith('prob_') for k in result.keys()):
                print("\nProbabilities:")
                for key, value in result.items():
                    if key.startswith('prob_'):
                        print(f"  {key.replace('prob_', '')}: {value:.2%}")
        
        except Exception as e:
            print(f"Error making prediction: {e}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_prediction()
    else:
        demo_predictions()

