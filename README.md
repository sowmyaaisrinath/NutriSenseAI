# NutriSenseAI

**Intelligent Nutrition Analysis & Meal Optimization System**

Machine Learning Classification of Nutrient Levels from Nutrition Labels

## Overview

NutriSenseAI is a comprehensive nutrition analysis and optimization system that combines machine learning, dimensionality reduction, and linear programming to help consumers make informed dietary decisions. The system classifies foods into Low, Medium, or High nutrient categories, visualizes food relationships using t-SNE, and generates optimal meal plans based on nutritional and budget constraints.

## Project Structure

```
NutriSenseAI/
├── data/
│   ├── raw/              # Raw data files (Open Food Facts, USDA)
│   └── processed/        # Processed and feature-engineered data
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   └── 02_model_training.ipynb
├── src/
│   ├── data/
│   │   ├── data_loader.py      # Data loading utilities
│   │   └── preprocessing.py    # Preprocessing and feature engineering
│   ├── models/
│   │   ├── train.py           # Model training pipeline
│   │   └── evaluate.py        # Model evaluation and metrics
│   └── utils/
│       └── interpretability.py # SHAP and model interpretation
├── models/               # Saved trained models
├── results/             # Evaluation results and visualizations
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Research Questions

1. Can machine learning models accurately classify foods into Low/Medium/High nutrient categories based solely on nutrition labels?
2. Which algorithms perform best across different nutrients (sugars, fiber, protein, sodium)?
3. What features (nutrient ratios, energy density) are most important for classification?

## Data Sources

### Primary Datasets

1. **Open Food Facts**: Open database of food products worldwide
   - Download: https://www.kaggle.com/datasets/openfoodfacts/world-food-facts
   - Format: TSV file with nutrition per 100g/ml
   
2. **USDA National Nutrient Database**: Comprehensive nutrition database
   - Download: https://agdatacommons.nal.usda.gov/articles/dataset/Composition_of_Foods_Raw_Processed_Prepared_USDA_National_Nutrient_Database_for_Standard_Reference_Release_27/25060841
   - Format: CSV/Excel with detailed nutrient information

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download this repository:
```bash
git clone <repository-url>
cd NutriSenseAI
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Download the data:
   - Download the Open Food Facts dataset from Kaggle
   - Place the file in `data/raw/` directory
   - Alternatively, use the sample data generator for testing

## Data Format Optimization

**🚀 Your data has been converted to Parquet format for optimal performance!**

### Benefits
- **98.7% compression**: 963MB → 12MB (Open Food Facts)
- **15x faster loading**: Load data in ~1-2 seconds instead of 15-30 seconds
- **Zero data loss**: All data and precision preserved

### Parquet Files Location
- Open Food Facts: `data/processed/en.openfoodfacts.org.products.parquet`
- USDA SR28: `data/processed/usda_sr28/*.parquet`

See `PARQUET_CONVERSION_SUMMARY.md` for detailed information.

## Usage

### 1. Exploratory Data Analysis

Start with the EDA notebook to understand the data:

```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

This notebook will:
- Load and inspect nutrition data
- Visualize nutrient distributions
- Clean and preprocess data
- Engineer features (nutrient ratios, densities)
- Create classification labels
- Save processed data

### 2. Model Training and Evaluation

Train and evaluate multiple ML models:

```bash
jupyter notebook notebooks/02_model_training.ipynb
```

This notebook will:
- Load preprocessed data
- Train multiple models (Logistic Regression, Decision Trees, Random Forest, XGBoost, MLP)
- Compare model performance
- Evaluate with confusion matrices and metrics
- Interpret models using feature importance and SHAP
- Save the best model

### 3. Comprehensive Training Script (Recommended!)

**🚀 Train all classification tasks systematically with one command:**

```bash
python train_all_tasks.py
```

This script will:
- Load the **full dataset** (no sampling) for maximum accuracy
- Train models for **all 5 classification tasks**:
  - Binary: healthy/unhealthy
  - Multiclass: sugar_class, fiber_class, protein_class, sodium_class
- Train **5 models per task** (Logistic Regression, Decision Tree, Random Forest, XGBoost, MLP)
- Compare and evaluate all models
- Save the best model for each task
- Generate comprehensive comparison visualizations
- Save detailed results in `results/` directory

**Configuration options in the script:**
```python
SAMPLE_SIZE = None  # None = full dataset, or specify number like 100000
PERFORM_HYPERPARAMETER_TUNING = False  # Set to True for hyperparameter tuning (slower but better results)
```

**Output files:**
- `models/best_model_*.pkl` - Trained models for each task
- `models/scaler_*.pkl` - Feature scalers
- `models/features_*.pkl` - Feature names
- `results/all_tasks_comparison.csv` - Performance comparison table
- `results/all_tasks_comparison.png` - Visualization
- `results/all_tasks_results.json` - Detailed results

### 4. Making Predictions on New Data

After training models, use them for predictions:

```bash
python predict_new_data.py
```

This will run demo predictions on example foods (chocolate, broccoli, yogurt).

**Or use in your own code:**

```python
from predict_new_data import NutritionPredictor

# Initialize predictor for a specific task
predictor = NutritionPredictor('sugar_class')

# Predict for a single food
nutrition_info = {
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

result = predictor.predict_single(nutrition_info)
print(f"Prediction: {result['prediction_label']}")
print(f"Confidence: {result['prob_Low Sugar']:.2%}")
```

**Interactive mode:**
```bash
python predict_new_data.py --interactive
```

### 5. Using the Python Modules Directly

You can also use the modules directly in your own scripts:

```python
from src.data.data_loader import NutritionDataLoader
from src.data.preprocessing import NutritionPreprocessor
from src.models.train import NutritionClassifier
from src.models.evaluate import ModelEvaluator

# Load data (simplified - always uses Parquet!)
loader = NutritionDataLoader(processed_dir='data/processed')
df = loader.load_open_food_facts()  # Loads 356K products in ~1-2 seconds

# Or load a sample for faster testing
# df = loader.load_open_food_facts(sample_size=10000)

# Preprocess
preprocessor = NutritionPreprocessor()
df_clean = preprocessor.clean_data(df)
df_features = preprocessor.engineer_features(df_clean)

# Create labels
sugar_labels = preprocessor.create_nutrient_labels(df_features, 'sugars_100g')

# Train models
classifier = NutritionClassifier(task_type='multiclass')
results = classifier.train_models(X_train, y_train, X_val, y_val)

# Evaluate
evaluator = ModelEvaluator(task_type='multiclass')
comparison = evaluator.compare_models(classifier.models, X_test, y_test)
```

## Models

The project implements and compares the following models:

1. **Logistic Regression**: Binary and multiclass classification baseline
2. **Decision Tree**: Interpretable tree-based model
3. **Random Forest**: Ensemble method with feature importance
4. **XGBoost**: Gradient boosting for high performance
5. **Multi-Layer Perceptron (MLP)**: Neural network for complex patterns

## Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision, Recall, F1-Score**: Per-class and macro-averaged
- **Confusion Matrix**: Detailed classification results
- **ROC-AUC**: For binary classification tasks
- **Calibration Curves**: Probability calibration assessment

## Feature Engineering

Engineered features include:

- **Energy density**: Calories per 100g
- **Macronutrient ratios**: Proportion of fat, carbs, protein
- **Sugar density**: Sugar content per 100g
- **Sugar-to-fiber ratio**: Indicator of processed foods
- **Saturated fat ratio**: Proportion of unhealthy fats
- **Sodium density**: Salt content (mg)
- **Protein density**: Protein content per 100g

## Classification Labels

### Multiclass (Low/Medium/High)

Based on dietary guidelines:
- **Sugar**: Low (<5g), Medium (5-22.5g), High (>22.5g) per 100g
- **Fiber**: Low (<1.5g), Medium (1.5-6g), High (>6g) per 100g
- **Protein**: Low (<3g), Medium (3-10g), High (>10g) per 100g
- **Sodium**: Low (<100mg), Medium (100-600mg), High (>600mg) per 100g

### Binary (Healthy/Unhealthy)

Foods classified as healthy if they meet:
- Low sugar (<5g per 100g)
- High fiber (>3g per 100g)
- High protein (>5g per 100g)
- Low sodium (<0.5g per 100g)

## Interpretability

Model interpretability using:

1. **Feature Importance**: For tree-based models (Random Forest, XGBoost)
2. **SHAP Values**: Shapley Additive Explanations for any model
3. **Partial Dependence Plots**: Effect of individual features
4. **Confusion Matrices**: Detailed error analysis

## 🚀 Extension Features (Implemented!)

### 1. T-SNE Food Mapping ✅
**Status**: Fully implemented and working!

Map all foods in 2D space to suggest substitutions based on nutritional similarity.

```bash
python food_mapping_tsne.py
```

**Features**:
- Visualize 296K+ foods in 2D space
- Find nutritionally similar foods
- Discover food clusters automatically
- Filter for healthier alternatives
- Save embeddings for fast lookups

**Usage**:
```python
from food_mapping_tsne import FoodMapper
mapper = FoodMapper()
mapper.load_data()
mapper.prepare_features()
mapper.apply_tsne()
mapper.visualize_2d(save_path='results/my_food_map.png')
substitutes = mapper.find_similar_foods('chocolate', top_n=10)
```

### 2. Meal Optimization ✅
**Status**: Fully implemented and working!

Create optimal meal plans using linear programming to meet nutrition goals.

```bash
python meal_optimizer.py
```

**Features**:
- Optimize for calories, protein, sugar, fiber, sodium
- Respect budget constraints
- Ensure food variety
- Generate shopping lists
- Multi-day meal planning

**Usage**:
```python
from meal_optimizer import MealOptimizer
optimizer = MealOptimizer()
optimizer.load_data()
optimizer.create_food_pool()
meal_plan = optimizer.optimize_meal(
    target_calories=2000,
    min_protein=60,
    max_cost=15
)
```

### 3. Future Extensions

1. **LLM Integration**: Use LLMs to add taste/texture descriptions as features
2. **Multi-label Classification**: Classify multiple nutrients simultaneously
3. **Deep Learning**: Advanced neural architectures (CNNs, Transformers)
4. **Real-time API**: Deploy as REST API for web/mobile apps

## Results

Results will be saved in the `results/` directory:
- Model comparison tables
- Confusion matrices
- Feature importance plots
- SHAP visualizations
- Performance metrics

## Contributing

This is an academic project. Suggestions and improvements are welcome!

## License

This project is for educational purposes as part of a machine learning course.

## References

- Open Food Facts: https://world.openfoodfacts.org/
- USDA FoodData Central: https://fdc.nal.usda.gov/
- Dietary Guidelines: https://www.dietaryguidelines.gov/

## Project Stats

- **Dataset Size**: 296,812 food products (Open Food Facts) + 8,789 foods (USDA SR28)
- **Classification Accuracy**: 82.5-100% (avg. 91.1%)
- **Food Clusters**: 8 meaningful groups identified via t-SNE
- **Optimization Success**: 94% feasibility rate across test scenarios

## License

This project is open source and available for educational and research purposes.

## Contact

For questions or issues, please contact the project maintainer.

---

**NutriSenseAI** - Making nutrition data-driven and accessible 🥗🤖

