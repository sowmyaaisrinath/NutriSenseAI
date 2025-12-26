"""
Data loading utilities for nutrition datasets (Parquet format)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import warnings

warnings.filterwarnings('ignore')


class NutritionDataLoader:
    """
    Load nutrition data from Parquet files for fast processing
    """
    
    def __init__(self, processed_dir: str = "../../data/processed"):
        """
        Initialize data loader
        
        Args:
            processed_dir: Directory containing processed Parquet files
        """
        self.processed_dir = Path(processed_dir)
        print(f"self.processed_dir: {self.processed_dir}")
        if not self.processed_dir.exists():
            raise FileNotFoundError(
                f"Processed data directory not found: {self.processed_dir}\n"
                f"Please run: python convert_to_parquet.py"
            )
        
    def load_open_food_facts(self, sample_size: Optional[int] = None, 
                            columns: Optional[list] = None) -> pd.DataFrame:
        """
        Load Open Food Facts dataset from Parquet
        
        Args:
            sample_size: Optional number of rows to sample (for faster testing)
            columns: Optional list of specific columns to load (faster)
            
        Returns:
            DataFrame with Open Food Facts data
        """
        parquet_path = self.processed_dir / "openfoodfactsproducts.parquet"
        print(f"parquet_path: {parquet_path}")
        print(f"parquet_path.exists(): {parquet_path.exists()}")
        
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Open Food Facts Parquet file not found: {parquet_path}\n"
                f"Please run: python convert_to_parquet.py"
            )
        
        print(f"Loading Open Food Facts from Parquet...")
        
        try:
            df = pd.read_parquet(parquet_path, columns=columns)
            
            if sample_size and sample_size < len(df):
                df = df.sample(n=sample_size, random_state=42)
            
            print(f"✓ Loaded {len(df):,} products")
            return df
            
        except Exception as e:
            print(f"Error loading Parquet file: {e}")
            raise
    
    def load_usda_table(self, table: str = 'FOOD_DES') -> pd.DataFrame:
        """
        Load a specific USDA SR28 table from Parquet
        
        Args:
            table: Which USDA table to load (FOOD_DES, NUT_DATA, NUTR_DEF, etc.)
            
        Returns:
            DataFrame with USDA data
        """
        parquet_path = self.processed_dir / "usda_sr28" / f"{table}.parquet"
        
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"USDA {table} Parquet file not found: {parquet_path}\n"
                f"Please run: python convert_to_parquet.py"
            )
        
        print(f"Loading USDA {table} from Parquet...")
        
        try:
            df = pd.read_parquet(parquet_path)
            print(f"✓ Loaded {len(df):,} rows")
            return df
            
        except Exception as e:
            print(f"Error loading USDA table: {e}")
            raise
    
    def load_usda_merged(self) -> pd.DataFrame:
        """
        Load and merge key USDA SR28 tables into a single DataFrame
        with food descriptions and all nutrients as columns
        
        Returns:
            Merged DataFrame with food descriptions and nutrient data
        """
        print("\nLoading and merging USDA SR28 tables...")
        
        # Load key tables
        food_des = self.load_usda_table('FOOD_DES')
        nut_data = self.load_usda_table('NUT_DATA')
        nutr_def = self.load_usda_table('NUTR_DEF')
        
        # Merge nutrient data with nutrient definitions
        print("Merging nutrient values with definitions...")
        nut_data_merged = nut_data.merge(nutr_def, on='Nutr_No', how='left')
        
        # Pivot to get nutrients as columns
        print("Pivoting nutrients to columns...")
        nut_pivot = nut_data_merged.pivot_table(
            index='NDB_No',
            columns='NutrDesc',
            values='Nutr_Val',
            aggfunc='first'
        ).reset_index()
        
        # Merge with food descriptions
        print("Merging with food descriptions...")
        df_merged = food_des.merge(nut_pivot, on='NDB_No', how='left')
        
        print(f"✓ Merged: {len(df_merged):,} foods with {len(df_merged.columns)} columns")
        return df_merged
    
    def get_sample_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate sample data for testing purposes
        
        Args:
            n_samples: Number of sample records to generate
            
        Returns:
            DataFrame with synthetic nutrition data
        """
        print(f"Generating {n_samples} sample records for testing...")
        
        np.random.seed(42)
        
        # Generate synthetic data
        data = {
            'product_name': [f'Product_{i}' for i in range(n_samples)],
            'energy_100g': np.random.uniform(50, 600, n_samples),
            'fat_100g': np.random.uniform(0, 40, n_samples),
            'saturated-fat_100g': np.random.uniform(0, 20, n_samples),
            'carbohydrates_100g': np.random.uniform(0, 80, n_samples),
            'sugars_100g': np.random.uniform(0, 50, n_samples),
            'fiber_100g': np.random.uniform(0, 15, n_samples),
            'proteins_100g': np.random.uniform(0, 30, n_samples),
            'sodium_100g': np.random.uniform(0, 2, n_samples),
            'salt_100g': np.random.uniform(0, 5, n_samples),
        }
        
        df = pd.DataFrame(data)
        print("Sample data generated successfully")
        return df


def inspect_data(df: pd.DataFrame, show_stats: bool = True) -> None:
    """
    Print basic information about the dataset
    
    Args:
        df: DataFrame to inspect
        show_stats: Whether to show detailed statistics
    """
    print("\n" + "="*70)
    print("DATASET OVERVIEW")
    print("="*70)
    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nData types:\n{df.dtypes}")
    
    missing = df.isnull().sum()
    if missing.sum() > 0:
        missing_pct = (missing / len(df) * 100).round(1)
        print(f"\nMissing values:")
        for col in missing[missing > 0].index:
            print(f"  {col}: {missing[col]:,} ({missing_pct[col]}%)")
    else:
        print("\nMissing values: None")
    
    if show_stats:
        print(f"\nBasic statistics:\n{df.describe()}")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Example usage
    print("="*70)
    print("NUTRITION DATA LOADER - EXAMPLE")
    print("="*70)
    
    try:
        loader = NutritionDataLoader()
        
        # Example 1: Load Open Food Facts
        print("\n1. Loading Open Food Facts...")
        df_off = loader.load_open_food_facts(sample_size=1000)
        inspect_data(df_off, show_stats=False)
        
        # Example 2: Load specific USDA table
        print("\n2. Loading USDA Food Descriptions...")
        df_food = loader.load_usda_table('FOOD_DES')
        print(f"Sample foods:\n{df_food[['NDB_No', 'Long_Desc']].head()}")
        
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\nPlease run the conversion script first:")
        print("  python convert_to_parquet.py")

