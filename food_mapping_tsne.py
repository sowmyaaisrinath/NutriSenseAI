"""
T-SNE Food Mapping and Substitution Finder
===========================================
Visualize foods in 2D space based on nutritional similarity and find healthy substitutes.

Features:
- 2D/3D visualization of food space
- Find nutritionally similar foods
- Interactive plots with product details
- Food substitution recommendations
- Cluster analysis
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('.')

from src.data.data_loader import NutritionDataLoader
from src.data.preprocessing import NutritionPreprocessor


class FoodMapper:
    """Map foods in 2D/3D space using t-SNE for visualization and substitutions."""
    
    def __init__(self, processed_dir='data/processed'):
        """Initialize the food mapper."""
        self.processed_dir = processed_dir
        self.df = None
        self.X_scaled = None
        self.X_embedded = None
        self.scaler = None
        self.feature_cols = None
        
    def load_data(self, sample_size=None):
        """Load and prepare nutrition data."""
        print("Loading nutrition data...")
        loader = NutritionDataLoader(processed_dir=self.processed_dir)
        
        # Load Open Food Facts
        df = loader.load_open_food_facts(sample_size=sample_size)
        print(f"✓ Loaded {len(df):,} products")
        
        # Preprocess
        preprocessor = NutritionPreprocessor()
        df = preprocessor.clean_data(df, drop_threshold=0.7)
        df = preprocessor.engineer_features(df)
        print(f"✓ After preprocessing: {len(df):,} products")
        
        self.df = df
        return df
    
    def prepare_features(self):
        """Prepare feature matrix for dimensionality reduction."""
        print("\nPreparing features...")
        
        # Select numeric nutrition features
        self.feature_cols = [
            'energy_100g', 'fat_100g', 'saturated-fat_100g',
            'carbohydrates_100g', 'sugars_100g', 'fiber_100g',
            'proteins_100g', 'sodium_100g',
            'energy_density', 'fat_ratio', 'carb_ratio', 'protein_ratio',
            'sugar_density', 'sugar_fiber_ratio', 'saturated_fat_ratio',
            'sodium_density', 'protein_density'
        ]
        
        # Extract features
        X = self.df[self.feature_cols].fillna(0)
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)
        
        print(f"✓ Prepared {self.X_scaled.shape[1]} features for {self.X_scaled.shape[0]:,} products")
        return self.X_scaled
    
    def apply_tsne(self, n_components=2, perplexity=30, use_pca=True, n_pca_components=50):
        """
        Apply t-SNE dimensionality reduction.
        
        Parameters:
        -----------
        n_components : int
            Number of dimensions (2 or 3)
        perplexity : int
            t-SNE perplexity parameter
        use_pca : bool
            Whether to apply PCA before t-SNE (recommended for large datasets)
        n_pca_components : int
            Number of PCA components to use
        """
        print(f"\nApplying dimensionality reduction...")
        
        X = self.X_scaled
        
        # Apply PCA first for speed (optional but recommended)
        if use_pca and X.shape[1] > n_pca_components:
            print(f"  Step 1: PCA to {n_pca_components} components...")
            pca = PCA(n_components=n_pca_components, random_state=42)
            X = pca.fit_transform(X)
            explained_var = pca.explained_variance_ratio_.sum()
            print(f"  ✓ PCA explains {explained_var:.1%} of variance")
        
        # Apply t-SNE
        print(f"  Step 2: t-SNE to {n_components} components...")
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=42,
            max_iter=1000,
            verbose=0
        )
        self.X_embedded = tsne.fit_transform(X)
        
        print(f"✓ Dimensionality reduction complete: {self.X_embedded.shape}")
        return self.X_embedded
    
    def visualize_2d(self, color_by='nutrition_grade_fr', save_path=None, figsize=(14, 10)):
        """
        Create 2D visualization of food space.
        
        Parameters:
        -----------
        color_by : str
            Column to color points by ('nutrition_grade_fr', 'sugar_class', etc.)
        save_path : str
            Path to save the figure
        figsize : tuple
            Figure size
        """
        if self.X_embedded is None or self.X_embedded.shape[1] < 2:
            raise ValueError("Run apply_tsne() first with n_components=2")
        
        print(f"\nCreating 2D visualization colored by '{color_by}'...")
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Colored by specified column
        if color_by in self.df.columns:
            # Map categories to numbers
            unique_vals = self.df[color_by].dropna().unique()
            color_map = {val: i for i, val in enumerate(unique_vals)}
            colors = self.df[color_by].map(color_map)
            
            scatter = axes[0].scatter(
                self.X_embedded[:, 0],
                self.X_embedded[:, 1],
                c=colors,
                cmap='RdYlGn_r' if 'grade' in color_by else 'viridis',
                alpha=0.6,
                s=20,
                edgecolors='none'
            )
            
            # Add colorbar with labels
            cbar = plt.colorbar(scatter, ax=axes[0])
            if len(unique_vals) <= 10:
                cbar.set_ticks(range(len(unique_vals)))
                cbar.set_ticklabels(unique_vals)
            
            axes[0].set_title(f'Food Products Colored by {color_by}', fontweight='bold', fontsize=14)
        else:
            axes[0].scatter(
                self.X_embedded[:, 0],
                self.X_embedded[:, 1],
                alpha=0.6,
                s=20,
                c='steelblue',
                edgecolors='none'
            )
            axes[0].set_title('Food Products in Nutrition Space', fontweight='bold', fontsize=14)
        
        axes[0].set_xlabel('t-SNE Component 1', fontsize=12)
        axes[0].set_ylabel('t-SNE Component 2', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Density plot
        axes[1].hexbin(
            self.X_embedded[:, 0],
            self.X_embedded[:, 1],
            gridsize=50,
            cmap='YlOrRd',
            mincnt=1
        )
        axes[1].set_title('Food Density in Nutrition Space', fontweight='bold', fontsize=14)
        axes[1].set_xlabel('t-SNE Component 1', fontsize=12)
        axes[1].set_ylabel('t-SNE Component 2', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved visualization to {save_path}")
        
        plt.show()
        
        return fig
    
    def find_clusters(self, n_clusters=10):
        """
        Find clusters in the embedded space.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters to find
        """
        print(f"\nFinding {n_clusters} clusters...")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df['cluster'] = kmeans.fit_predict(self.X_embedded)
        
        # Analyze clusters
        print("\nCluster Statistics:")
        print("="*80)
        
        for i in range(n_clusters):
            cluster_df = self.df[self.df['cluster'] == i]
            print(f"\nCluster {i}: {len(cluster_df):,} products")
            
            # Most common categories
            if 'categories' in cluster_df.columns:
                top_cats = cluster_df['categories'].value_counts().head(3)
                print(f"  Top categories: {', '.join(top_cats.index[:3].tolist())}")
            
            # Average nutrition
            print(f"  Avg calories: {cluster_df['energy_100g'].mean():.0f} kcal/100g")
            print(f"  Avg sugar: {cluster_df['sugars_100g'].mean():.1f}g/100g")
            print(f"  Avg protein: {cluster_df['proteins_100g'].mean():.1f}g/100g")
        
        return self.df['cluster']
    
    def find_similar_foods(self, food_name=None, food_idx=None, top_n=10, 
                          same_category=False, healthier_only=False):
        """
        Find nutritionally similar foods.
        
        Parameters:
        -----------
        food_name : str
            Name of food to find substitutes for
        food_idx : int
            Index of food in dataframe
        top_n : int
            Number of similar foods to return
        same_category : bool
            Only return foods from same category
        healthier_only : bool
            Only return healthier alternatives
        """
        # Find the target food
        if food_idx is None:
            if food_name is None:
                raise ValueError("Must provide either food_name or food_idx")
            
            # Search for food by name (case-insensitive, partial match)
            mask = self.df['product_name'].str.contains(food_name, case=False, na=False)
            matches = self.df[mask]
            
            if len(matches) == 0:
                print(f"No foods found matching '{food_name}'")
                return None
            elif len(matches) > 1:
                print(f"Found {len(matches)} matches for '{food_name}'. Using first match.")
            
            # Get the positional index (iloc) not the label index (loc)
            food_idx = self.df.index.get_loc(matches.index[0])
        elif food_idx >= len(self.df):
            # If provided idx is out of range, convert from label to position
            food_idx = self.df.index.get_loc(food_idx)
        
        # Get target food info
        target_food = self.df.iloc[food_idx]
        print(f"\nFinding substitutes for: {target_food['product_name']}")
        
        # Calculate distances in embedded space
        point = self.X_embedded[food_idx].reshape(1, -1)
        distances = cdist(point, self.X_embedded, metric='euclidean')[0]
        
        # Get candidates (excluding the food itself)
        candidate_indices = np.where(np.arange(len(self.df)) != food_idx)[0]
        candidate_distances = distances[candidate_indices]
        
        # Filter by category if requested
        if same_category and 'categories' in self.df.columns:
            target_category = target_food.get('categories')
            if pd.notna(target_category):
                category_mask = self.df.iloc[candidate_indices]['categories'].values == target_category
                valid_candidates = candidate_indices[category_mask]
                valid_distances = candidate_distances[category_mask]
            else:
                valid_candidates = candidate_indices
                valid_distances = candidate_distances
        else:
            valid_candidates = candidate_indices
            valid_distances = candidate_distances
        
        # Filter by health if requested
        if healthier_only and 'nutrition_grade_fr' in self.df.columns:
            target_grade = target_food.get('nutrition_grade_fr')
            if pd.notna(target_grade):
                grade_order = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
                target_grade_val = grade_order.get(target_grade, 5)
                
                def is_healthier(grade):
                    if pd.isna(grade):
                        return False
                    return grade_order.get(grade, 5) < target_grade_val
                
                health_mask = self.df.iloc[valid_candidates]['nutrition_grade_fr'].apply(is_healthier).values
                valid_candidates = valid_candidates[health_mask]
                valid_distances = valid_distances[health_mask]
        
        # Get top N nearest
        if len(valid_candidates) == 0:
            print("No valid substitutes found with the given criteria.")
            return None
        
        nearest_idx = np.argsort(valid_distances)[:top_n]
        similar_indices = valid_candidates[nearest_idx]
        similar_distances = valid_distances[nearest_idx]
        
        # Create results dataframe
        result_cols = ['product_name', 'brands', 'nutrition_grade_fr',
                      'energy_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g']
        available_cols = [col for col in result_cols if col in self.df.columns]
        
        results = self.df.iloc[similar_indices][available_cols].copy()
        results['similarity_score'] = 1 / (1 + similar_distances)  # Convert distance to similarity
        results['distance'] = similar_distances
        
        print(f"\n{'='*80}")
        print(f"TOP {len(results)} SIMILAR FOODS")
        print(f"{'='*80}")
        print(results.to_string(index=False))
        
        return results
    
    def save_embeddings(self, output_path='data/processed/food_embeddings.parquet'):
        """Save embeddings and metadata for later use."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create dataframe with embeddings
        embedding_df = pd.DataFrame(
            self.X_embedded,
            columns=[f'tsne_{i}' for i in range(self.X_embedded.shape[1])],
            index=self.df.index
        )
        
        # Combine with original data
        combined = pd.concat([self.df, embedding_df], axis=1)
        
        # Save
        combined.to_parquet(output_path)
        print(f"\n✓ Saved embeddings to {output_path}")
        return output_path


def demo():
    """Run demonstration of food mapping features."""
    print("="*80)
    print("T-SNE FOOD MAPPING DEMO")
    print("="*80)
    
    # Initialize mapper
    mapper = FoodMapper(processed_dir='data/processed')
    
    # Load data (use sample for demo speed)
    print("\nStep 1: Loading data...")
    mapper.load_data(sample_size=10000)  # Use 10K for demo; set to None for full dataset
    
    # Prepare features
    print("\nStep 2: Preparing features...")
    mapper.prepare_features()
    
    # Apply t-SNE
    print("\nStep 3: Applying t-SNE...")
    mapper.apply_tsne(n_components=2, perplexity=30)
    
    # Visualize
    print("\nStep 4: Creating visualizations...")
    mapper.visualize_2d(
        color_by='nutrition_grade_fr',
        save_path='results/food_map_tsne.png'
    )
    
    # Find clusters
    print("\nStep 5: Finding clusters...")
    mapper.find_clusters(n_clusters=8)
    
    # Find similar foods - Example 1: Chocolate
    print("\n" + "="*80)
    print("EXAMPLE 1: Finding substitutes for chocolate")
    print("="*80)
    mapper.find_similar_foods('chocolate', top_n=10)
    
    # Find similar foods - Example 2: Yogurt (healthier alternatives)
    print("\n" + "="*80)
    print("EXAMPLE 2: Finding HEALTHIER alternatives to yogurt")
    print("="*80)
    mapper.find_similar_foods('yogurt', top_n=10, healthier_only=True)
    
    # Find similar foods - Example 3: Pizza (same category)
    print("\n" + "="*80)
    print("EXAMPLE 3: Finding similar foods to pizza (same category)")
    print("="*80)
    mapper.find_similar_foods('pizza', top_n=10, same_category=True)
    
    # Save embeddings
    print("\nStep 6: Saving embeddings...")
    mapper.save_embeddings()
    
    print("\n" + "="*80)
    print("✅ DEMO COMPLETE!")
    print("="*80)
    print("\n📁 Output files:")
    print("  - results/food_map_tsne.png - Visualization")
    print("  - data/processed/food_embeddings.parquet - Saved embeddings")
    print("\n💡 To use in your own code:")
    print("```python")
    print("from food_mapping_tsne import FoodMapper")
    print("mapper = FoodMapper()")
    print("mapper.load_data()")
    print("mapper.prepare_features()")
    print("mapper.apply_tsne()")
    print("substitutes = mapper.find_similar_foods('your_food_name')")
    print("```")


if __name__ == '__main__':
    demo()

