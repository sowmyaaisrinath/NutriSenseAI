"""
Data preprocessing and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List, Optional, Dict
import warnings

warnings.filterwarnings('ignore')


class NutritionPreprocessor:
    """
    Preprocess nutrition data and engineer features
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.nutrient_columns = [
            'energy_100g', 'fat_100g', 'saturated-fat_100g',
            'carbohydrates_100g', 'sugars_100g', 'fiber_100g',
            'proteins_100g', 'sodium_100g', 'salt_100g'
        ]
    
    def clean_data(self, df: pd.DataFrame, drop_threshold: float = 0.5) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and outliers
        
        Args:
            df: Input DataFrame
            drop_threshold: Drop columns with more than this fraction of missing values
            
        Returns:
            Cleaned DataFrame
        """
        print("Cleaning data...")
        df_clean = df.copy()
        
        # Remove rows with all missing nutrient values
        nutrient_cols = [col for col in self.nutrient_columns if col in df_clean.columns]
        df_clean = df_clean.dropna(subset=nutrient_cols, how='all')
        
        # Drop columns with too many missing values
        missing_fraction = df_clean.isnull().sum() / len(df_clean)
        cols_to_keep = missing_fraction[missing_fraction < drop_threshold].index.tolist()
        df_clean = df_clean[cols_to_keep]
        
        # Fill remaining missing values with median for numeric columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        # Remove negative values (data errors)
        for col in numeric_cols:
            if col in nutrient_cols:
                df_clean[col] = df_clean[col].clip(lower=0)
        
        # Remove extreme outliers (values beyond 99th percentile)
        for col in nutrient_cols:
            if col in df_clean.columns:
                threshold = df_clean[col].quantile(0.99)
                df_clean[col] = df_clean[col].clip(upper=threshold)
        
        print(f"Data cleaned: {len(df)} -> {len(df_clean)} rows")
        return df_clean
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features from raw nutrition data
        
        Args:
            df: Input DataFrame with nutrition data
            
        Returns:
            DataFrame with additional engineered features
        """
        print("Engineering features...")
        df_feat = df.copy()
        
        # Energy density (calories per 100g)
        if 'energy_100g' in df_feat.columns:
            df_feat['energy_density'] = df_feat['energy_100g']
        
        # Macronutrient ratios
        if all(col in df_feat.columns for col in ['fat_100g', 'carbohydrates_100g', 'proteins_100g']):
            total_macros = df_feat['fat_100g'] + df_feat['carbohydrates_100g'] + df_feat['proteins_100g']
            total_macros = total_macros.replace(0, 1)  # Avoid division by zero
            
            df_feat['fat_ratio'] = df_feat['fat_100g'] / total_macros
            df_feat['carb_ratio'] = df_feat['carbohydrates_100g'] / total_macros
            df_feat['protein_ratio'] = df_feat['proteins_100g'] / total_macros
        
        # Sugar density (sugars per 100g)
        if 'sugars_100g' in df_feat.columns:
            df_feat['sugar_density'] = df_feat['sugars_100g']
        
        # Sugar to fiber ratio (higher is worse)
        if all(col in df_feat.columns for col in ['sugars_100g', 'fiber_100g']):
            fiber_safe = df_feat['fiber_100g'].replace(0, 0.1)
            df_feat['sugar_fiber_ratio'] = df_feat['sugars_100g'] / fiber_safe
        
        # Saturated fat ratio
        if all(col in df_feat.columns for col in ['saturated-fat_100g', 'fat_100g']):
            fat_safe = df_feat['fat_100g'].replace(0, 1)
            df_feat['saturated_fat_ratio'] = df_feat['saturated-fat_100g'] / fat_safe
        
        # Sodium density
        if 'sodium_100g' in df_feat.columns:
            df_feat['sodium_density'] = df_feat['sodium_100g'] * 1000  # Convert to mg
        
        # Protein density
        if 'proteins_100g' in df_feat.columns:
            df_feat['protein_density'] = df_feat['proteins_100g']
        
        print(f"Features engineered: {df.shape[1]} -> {df_feat.shape[1]} columns")
        return df_feat
    
    def get_leaky_features(self, task: str) -> list:
        """
        Get list of features that would cause data leakage for a given task.
        
        Parameters:
        -----------
        task : str
            Task name ('sugar_class', 'protein_class', 'fiber_class', 'sodium_class', 'healthy')
            
        Returns:
        --------
        list : Features to exclude to prevent data leakage
        """
        leaky_features_map = {
            'sugar_class': ['sugars_100g', 'sugar_density', 'sugar_fiber_ratio'],
            'protein_class': ['proteins_100g', 'protein_ratio', 'protein_density'],
            'fiber_class': ['fiber_100g', 'sugar_fiber_ratio'],
            'sodium_class': ['sodium_100g', 'sodium_density', 'salt_100g'],
            'healthy': []  # Binary task uses multiple features, less direct leakage
        }
        return leaky_features_map.get(task, [])
    
    def create_nutrient_labels(self, df: pd.DataFrame, nutrient: str, 
                               thresholds: Optional[Dict[str, float]] = None) -> pd.Series:
        """
        Create Low/Medium/High classification labels for a nutrient
        
        Args:
            df: Input DataFrame
            nutrient: Name of nutrient column
            thresholds: Optional dict with 'low' and 'high' threshold values
                       If None, uses 33rd and 67th percentiles
            
        Returns:
            Series with classification labels (0=Low, 1=Medium, 2=High)
        """
        if nutrient not in df.columns:
            raise ValueError(f"Nutrient '{nutrient}' not found in DataFrame")
        
        values = df[nutrient]
        
        if thresholds is None:
            # Use percentile-based thresholds
            low_threshold = values.quantile(0.33)
            high_threshold = values.quantile(0.67)
        else:
            low_threshold = thresholds.get('low', values.quantile(0.33))
            high_threshold = thresholds.get('high', values.quantile(0.67))
        
        labels = pd.cut(values, 
                       bins=[-np.inf, low_threshold, high_threshold, np.inf],
                       labels=[0, 1, 2])  # 0=Low, 1=Medium, 2=High
        
        return labels
    
    def create_binary_labels(self, df: pd.DataFrame, 
                            healthy_criteria: Optional[Dict] = None) -> pd.Series:
        """
        Create binary Healthy/Unhealthy labels based on multiple criteria
        
        Args:
            df: Input DataFrame
            healthy_criteria: Dict with criteria for healthy classification
            
        Returns:
            Series with binary labels (0=Unhealthy, 1=Healthy)
        """
        if healthy_criteria is None:
            # Default criteria (can be adjusted based on dietary guidelines)
            healthy_criteria = {
                'sugars_100g': ('low', 5),  # Low sugar: < 5g per 100g
                'fiber_100g': ('high', 3),   # High fiber: > 3g per 100g
                'proteins_100g': ('high', 5), # High protein: > 5g per 100g
                'sodium_100g': ('low', 0.5),  # Low sodium: < 0.5g per 100g
            }
        
        # Start with all products as healthy (1)
        healthy = pd.Series(1, index=df.index)
        
        for nutrient, (direction, threshold) in healthy_criteria.items():
            if nutrient in df.columns:
                if direction == 'low':
                    # Mark as unhealthy if above threshold
                    healthy &= (df[nutrient] <= threshold)
                elif direction == 'high':
                    # Mark as unhealthy if below threshold
                    healthy &= (df[nutrient] >= threshold)
        
        return healthy.astype(int)
    
    def normalize_features(self, X_train: pd.DataFrame, 
                          X_test: Optional[pd.DataFrame] = None) -> Tuple:
        """
        Normalize features using StandardScaler
        
        Args:
            X_train: Training features
            X_test: Optional test features
            
        Returns:
            Tuple of normalized (X_train, X_test) or just X_train if X_test is None
        """
        print("Normalizing features...")
        
        # Fit scaler on training data
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled


# Dietary reference thresholds (based on general guidelines)
NUTRIENT_THRESHOLDS = {
    'sugars_100g': {'low': 5, 'high': 22.5},      # g per 100g
    'fiber_100g': {'low': 1.5, 'high': 6},         # g per 100g
    'proteins_100g': {'low': 3, 'high': 10},       # g per 100g
    'sodium_100g': {'low': 0.1, 'high': 0.6},      # g per 100g (0.1g = 100mg)
    'fat_100g': {'low': 3, 'high': 17.5},          # g per 100g
    'saturated-fat_100g': {'low': 1.5, 'high': 5}, # g per 100g
}


if __name__ == "__main__":
    # Example usage
    from data_loader import NutritionDataLoader
    
    # Load sample data
    loader = NutritionDataLoader()
    df = loader.get_sample_data(n_samples=100)
    
    # Preprocess
    preprocessor = NutritionPreprocessor()
    df_clean = preprocessor.clean_data(df)
    df_features = preprocessor.engineer_features(df_clean)
    
    # Create labels
    sugar_labels = preprocessor.create_nutrient_labels(df_features, 'sugars_100g')
    print(f"\nSugar labels distribution:\n{sugar_labels.value_counts()}")
    
    healthy_labels = preprocessor.create_binary_labels(df_features)
    print(f"\nHealthy/Unhealthy distribution:\n{healthy_labels.value_counts()}")

