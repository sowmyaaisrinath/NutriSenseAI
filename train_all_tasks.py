"""
Comprehensive Training Script - All Classification Tasks
=========================================================
This script systematically trains models for all classification tasks:
- Binary: healthy/unhealthy
- Multiclass: sugar_class, fiber_class, protein_class, sodium_class

Features:
- Full dataset training (no sampling)
- Hyperparameter tuning
- Cross-task comparison
- Model persistence
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime
import json

# Add project root to path
sys.path.append('.')

from src.data.data_loader import NutritionDataLoader
from src.data.preprocessing import NutritionPreprocessor
from src.models.train import NutritionClassifier
from src.models.evaluate import ModelEvaluator
from src.utils.interpretability import ModelInterpreter

import warnings
warnings.filterwarnings('ignore')

# Configuration
SAMPLE_SIZE = None  # None = use full dataset, or specify number like 100000
PERFORM_HYPERPARAMETER_TUNING = False  # Set to True for hyperparameter tuning (slower)
RANDOM_STATE = 42

# Define all tasks
TASKS = [
    {'name': 'healthy', 'type': 'binary', 'description': 'Healthy vs Unhealthy'},
    {'name': 'sugar_class', 'type': 'multiclass', 'description': 'Sugar content (Low/Medium/High)'},
    {'name': 'fiber_class', 'type': 'multiclass', 'description': 'Fiber content (Low/Medium/High)'},
    {'name': 'protein_class', 'type': 'multiclass', 'description': 'Protein content (Low/Medium/High)'},
    {'name': 'sodium_class', 'type': 'multiclass', 'description': 'Sodium content (Low/Medium/High)'}
]


def load_and_prepare_data(sample_size=None):
    """Load and preprocess data."""
    print("="*80)
    print("STEP 1: LOADING AND PREPARING DATA")
    print("="*80)
    
    loader = NutritionDataLoader(processed_dir='data/processed')
    
    # Load Open Food Facts data
    print(f"\nLoading Open Food Facts data...")
    if sample_size:
        print(f"  Using sample size: {sample_size:,}")
    else:
        print(f"  Using FULL dataset")
    
    df_raw = loader.load_open_food_facts(sample_size=sample_size)
    print(f"✓ Loaded {len(df_raw):,} products")
    
    # Initialize preprocessor
    preprocessor = NutritionPreprocessor()
    
    # Clean data
    print("\nCleaning data...")
    df_clean = preprocessor.clean_data(df_raw, drop_threshold=0.7)
    print(f"✓ Cleaned: {len(df_clean):,} products retained")
    
    # Engineer features
    print("\nEngineering features...")
    df_features = preprocessor.engineer_features(df_clean)
    print(f"✓ Features engineered: {df_features.shape[1]} total columns")
    
    # Create all labels
    print("\nCreating classification labels...")
    
    # Binary labels
    df_features['healthy'] = preprocessor.create_binary_labels(df_features)
    print(f"  ✓ Binary labels (healthy): {df_features['healthy'].value_counts().to_dict()}")
    
    # Multiclass labels
    from src.data.preprocessing import NUTRIENT_THRESHOLDS
    
    for nutrient, label_name in [
        ('sugars_100g', 'sugar_class'),
        ('fiber_100g', 'fiber_class'),
        ('proteins_100g', 'protein_class'),
        ('sodium_100g', 'sodium_class')
    ]:
        if nutrient in df_features.columns:
            df_features[label_name] = preprocessor.create_nutrient_labels(
                df_features, nutrient, NUTRIENT_THRESHOLDS.get(nutrient)
            )
            print(f"  ✓ Multiclass labels ({label_name}): {df_features[label_name].value_counts().to_dict()}")
    
    return df_features, preprocessor


def train_task(df_features, task_info, preprocessor, perform_tuning=False):
    """Train models for a specific task."""
    task_name = task_info['name']
    task_type = task_info['type']
    
    print("\n" + "="*80)
    print(f"STEP 2: TRAINING MODELS FOR TASK: {task_name.upper()}")
    print(f"Description: {task_info['description']}")
    print("="*80)
    
    # Define label and feature columns
    label_cols = ['healthy', 'sugar_class', 'fiber_class', 'protein_class', 'sodium_class',
                  'product_name', 'brands', 'categories', 'countries', 'nutrition_grade_fr']
    
    feature_cols = [col for col in df_features.columns if col not in label_cols]
    
    # IMPORTANT: Exclude leaky features to prevent data leakage
    leaky_features = preprocessor.get_leaky_features(task_name)
    if leaky_features:
        print(f"\n🔒 Preventing data leakage:")
        print(f"   Excluding features: {leaky_features}")
        feature_cols = [col for col in feature_cols if col not in leaky_features]
        print(f"   Features remaining: {len(feature_cols)}")
    
    # Prepare X and y
    X = df_features[feature_cols].copy()
    y = df_features[task_name].copy()
    
    # Remove samples with missing labels
    mask = ~y.isnull()
    X = X[mask]
    y = y[mask]
    
    print(f"\nDataset info:")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Samples: {len(X):,}")
    print(f"  Task type: {task_type}")
    print(f"  Number of classes: {y.nunique()}")
    print(f"  Label distribution: {y.value_counts().sort_index().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train
    )
    
    print(f"\nData splits:")
    print(f"  Training:   {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation: {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test:       {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")
    
    # Normalize features
    X_train_scaled, X_val_scaled = preprocessor.normalize_features(X_train, X_val)
    _, X_test_scaled = preprocessor.normalize_features(X_train, X_test)
    
    # Initialize classifier
    classifier = NutritionClassifier(task_type=task_type)
    
    # Train models
    print(f"\n{'='*80}")
    print("TRAINING MODELS...")
    print(f"{'='*80}")
    
    # First train with default parameters
    print("📊 Training with default parameters...")
    results = classifier.train_models(
        X_train_scaled, y_train,
        X_val_scaled, y_val
    )
    
    # Then perform hyperparameter tuning if requested
    if perform_tuning:
        print("\n⚙️  Performing hyperparameter tuning (this may take a while)...")
        from src.models.train import get_default_param_grids
        param_grids = get_default_param_grids()
        
        for model_name in classifier.models.keys():
            if model_name in param_grids:
                print(f"\nTuning {model_name}...")
                try:
                    classifier.hyperparameter_tuning(
                        model_name,
                        X_train_scaled,
                        y_train,
                        param_grids[model_name]
                    )
                except Exception as e:
                    print(f"  Warning: Tuning failed for {model_name}: {e}")
        
        # Re-evaluate after tuning
        print("\nRe-evaluating models after tuning...")
        results = classifier.train_models(
            X_train_scaled, y_train,
            X_val_scaled, y_val
        )
    
    print(f"\n✓ All models trained!")
    print(f"  Best model: {classifier.best_model_name}")
    
    # Evaluate on test set
    print(f"\n{'='*80}")
    print("EVALUATING MODELS ON TEST SET...")
    print(f"{'='*80}")
    
    evaluator = ModelEvaluator(task_type=task_type)
    comparison_df = evaluator.compare_models(classifier.models, X_test_scaled, y_test)
    
    print("\nModel Comparison:")
    print(comparison_df.round(4))
    
    # Find best model
    best_model_name = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']
    best_metrics = comparison_df[comparison_df['Model'] == best_model_name].iloc[0]
    
    print(f"\n🏆 Best model: {best_model_name}")
    print(f"   Accuracy:  {best_metrics['Accuracy']:.4f}")
    print(f"   F1-Score:  {best_metrics['F1-Score']:.4f}")
    print(f"   Precision: {best_metrics['Precision']:.4f}")
    print(f"   Recall:    {best_metrics['Recall']:.4f}")
    
    # Save results
    results_dict = {
        'task_name': task_name,
        'task_type': task_type,
        'best_model': best_model_name,
        'metrics': {
            'accuracy': float(best_metrics['Accuracy']),
            'f1_score': float(best_metrics['F1-Score']),
            'precision': float(best_metrics['Precision']),
            'recall': float(best_metrics['Recall'])
        },
        'all_models': comparison_df.to_dict('records'),
        'dataset_info': {
            'total_samples': len(X),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'num_features': len(feature_cols),
            'num_classes': int(y.nunique())
        },
        'timestamp': datetime.now().isoformat()
    }
    
    return {
        'results': results_dict,
        'classifier': classifier,
        'preprocessor': preprocessor,
        'evaluator': evaluator,
        'comparison_df': comparison_df,
        'feature_cols': feature_cols,
        'X_test_scaled': X_test_scaled,
        'y_test': y_test,
        'best_model_name': best_model_name
    }


def save_models(task_name, classifier, preprocessor, feature_cols, best_model_name):
    """Save trained models and artifacts."""
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Save best model
    model_path = models_dir / f'best_model_{task_name}.pkl'
    joblib.dump(classifier.models[best_model_name], model_path)
    
    # Save scaler
    scaler_path = models_dir / f'scaler_{task_name}.pkl'
    joblib.dump(preprocessor.scaler, scaler_path)
    
    # Save feature names
    features_path = models_dir / f'features_{task_name}.pkl'
    joblib.dump(feature_cols, features_path)
    
    print(f"\n✓ Saved artifacts:")
    print(f"  Model:    {model_path}")
    print(f"  Scaler:   {scaler_path}")
    print(f"  Features: {features_path}")


def compare_all_tasks(all_results):
    """Compare results across all tasks."""
    print("\n" + "="*80)
    print("STEP 3: CROSS-TASK COMPARISON")
    print("="*80)
    
    # Create comparison DataFrame
    comparison_data = []
    for result in all_results:
        r = result['results']
        comparison_data.append({
            'Task': r['task_name'],
            'Type': r['task_type'],
            'Best Model': r['best_model'],
            'Accuracy': r['metrics']['accuracy'],
            'F1-Score': r['metrics']['f1_score'],
            'Precision': r['metrics']['precision'],
            'Recall': r['metrics']['recall'],
            'Samples': r['dataset_info']['total_samples']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\nResults Summary:")
    print(comparison_df.round(4))
    
    # Save comparison
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    comparison_path = results_dir / 'all_tasks_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n✓ Saved comparison to: {comparison_path}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # F1-Score by task
    axes[0, 0].barh(comparison_df['Task'], comparison_df['F1-Score'], color='steelblue')
    axes[0, 0].set_xlabel('F1-Score')
    axes[0, 0].set_title('F1-Score by Task', fontweight='bold')
    axes[0, 0].set_xlim([0, 1])
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy by task
    axes[0, 1].barh(comparison_df['Task'], comparison_df['Accuracy'], color='coral')
    axes[0, 1].set_xlabel('Accuracy')
    axes[0, 1].set_title('Accuracy by Task', fontweight='bold')
    axes[0, 1].set_xlim([0, 1])
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision vs Recall
    axes[1, 0].scatter(comparison_df['Recall'], comparison_df['Precision'], 
                       s=200, alpha=0.6, color='green')
    for idx, row in comparison_df.iterrows():
        axes[1, 0].annotate(row['Task'], (row['Recall'], row['Precision']),
                           fontsize=8, ha='center')
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision vs Recall', fontweight='bold')
    axes[1, 0].set_xlim([0, 1])
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(True, alpha=0.3)
    
    # Best model distribution
    model_counts = comparison_df['Best Model'].value_counts()
    axes[1, 1].pie(model_counts.values, labels=model_counts.index, autopct='%1.0f%%',
                   startangle=90)
    axes[1, 1].set_title('Best Model Distribution', fontweight='bold')
    
    plt.tight_layout()
    plot_path = results_dir / 'all_tasks_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization to: {plot_path}")
    plt.close()
    
    # Save detailed results
    detailed_results = {
        'summary': comparison_df.to_dict('records'),
        'tasks': [r['results'] for r in all_results],
        'training_config': {
            'sample_size': SAMPLE_SIZE,
            'perform_hyperparameter_tuning': PERFORM_HYPERPARAMETER_TUNING,
            'random_state': RANDOM_STATE
        }
    }
    
    json_path = results_dir / 'all_tasks_results.json'
    with open(json_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    print(f"✓ Saved detailed results to: {json_path}")
    
    return comparison_df


def main():
    """Main training pipeline."""
    print("\n" + "🚀 "*20)
    print("COMPREHENSIVE NUTRITION CLASSIFICATION TRAINING")
    print("🚀 "*20)
    print(f"\nConfiguration:")
    print(f"  Sample size: {'FULL DATASET' if SAMPLE_SIZE is None else f'{SAMPLE_SIZE:,}'}")
    print(f"  Hyperparameter tuning: {'ENABLED' if PERFORM_HYPERPARAMETER_TUNING else 'DISABLED'}")
    print(f"  Number of tasks: {len(TASKS)}")
    print(f"  Tasks: {', '.join([t['name'] for t in TASKS])}")
    
    # Load and prepare data once
    df_features, preprocessor = load_and_prepare_data(sample_size=SAMPLE_SIZE)
    
    # Train all tasks
    all_results = []
    for i, task_info in enumerate(TASKS, 1):
        print(f"\n\n{'#'*80}")
        print(f"# TASK {i}/{len(TASKS)}: {task_info['name'].upper()}")
        print(f"{'#'*80}")
        
        try:
            result = train_task(df_features, task_info, preprocessor, 
                              perform_tuning=PERFORM_HYPERPARAMETER_TUNING)
            all_results.append(result)
            
            # Save models
            save_models(
                task_info['name'],
                result['classifier'],
                result['preprocessor'],
                result['feature_cols'],
                result['best_model_name']
            )
            
        except Exception as e:
            print(f"\n❌ Error training task {task_info['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Compare all tasks
    if all_results:
        comparison_df = compare_all_tasks(all_results)
        
        # Final summary
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE!")
        print("="*80)
        print(f"\nSuccessfully trained {len(all_results)}/{len(TASKS)} tasks")
        print("\nBest performing task:")
        best_task = comparison_df.loc[comparison_df['F1-Score'].idxmax()]
        print(f"  Task: {best_task['Task']}")
        print(f"  Model: {best_task['Best Model']}")
        print(f"  F1-Score: {best_task['F1-Score']:.4f}")
        print(f"  Accuracy: {best_task['Accuracy']:.4f}")
        
        print("\n📁 Output files:")
        print("  - models/best_model_*.pkl (trained models)")
        print("  - models/scaler_*.pkl (feature scalers)")
        print("  - models/features_*.pkl (feature names)")
        print("  - results/all_tasks_comparison.csv (comparison table)")
        print("  - results/all_tasks_comparison.png (visualization)")
        print("  - results/all_tasks_results.json (detailed results)")
    else:
        print("\n❌ No tasks were successfully trained!")


if __name__ == '__main__':
    main()

