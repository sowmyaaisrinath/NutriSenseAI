"""
Meal Optimization using Linear Programming
===========================================
Create optimal meal plans based on nutrition goals and constraints.

Features:
- Optimize meals for nutrition targets
- Respect budget constraints
- Handle dietary restrictions
- Generate shopping lists
- Multi-day meal planning
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('.')

from src.data.data_loader import NutritionDataLoader
from src.data.preprocessing import NutritionPreprocessor

# Try to import optimization libraries
try:
    from pulp import LpProblem, LpVariable, LpMinimize, LpMaximize, lpSum, value, LpStatus
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    print("⚠️  PuLP not installed. Run: pip install pulp")


class MealOptimizer:
    """Optimize meal plans using linear programming."""
    
    def __init__(self, processed_dir='data/processed'):
        """Initialize the meal optimizer."""
        if not PULP_AVAILABLE:
            raise ImportError("PuLP is required. Install with: pip install pulp")
        
        self.processed_dir = processed_dir
        self.df = None
        self.food_pool = None
        
    def load_data(self, sample_size=None):
        """Load and prepare food data."""
        print("Loading nutrition data...")
        loader = NutritionDataLoader(processed_dir=self.processed_dir)
        
        # Load Open Food Facts
        df = loader.load_open_food_facts(sample_size=sample_size)
        print(f"✓ Loaded {len(df):,} products")
        
        # Preprocess
        preprocessor = NutritionPreprocessor()
        df = preprocessor.clean_data(df, drop_threshold=0.5)
        print(f"✓ After preprocessing: {len(df):,} products")
        
        self.df = df
        return df
    
    def create_food_pool(self, filters=None, max_foods=500):
        """
        Create a pool of foods to optimize from.
        
        Parameters:
        -----------
        filters : dict
            Filters to apply (e.g., {'nutrition_grade_fr': ['a', 'b']})
        max_foods : int
            Maximum number of foods to include
        """
        df = self.df.copy()
        
        # Apply filters
        if filters:
            for column, values in filters.items():
                if column in df.columns:
                    df = df[df[column].isin(values)]
        
        # Remove duplicates and invalid entries
        df = df.dropna(subset=['energy_100g', 'proteins_100g'])
        
        # Sample if too many foods
        if len(df) > max_foods:
            df = df.sample(max_foods, random_state=42)
        
        # Add estimated cost (since cost isn't in the data)
        # This is a rough estimate based on food categories
        df['cost_per_100g'] = self._estimate_cost(df)
        
        self.food_pool = df.reset_index(drop=True)
        print(f"✓ Created food pool with {len(self.food_pool):,} foods")
        
        return self.food_pool
    
    def _estimate_cost(self, df):
        """Estimate cost per 100g based on food characteristics."""
        # Simple heuristic: higher protein/quality = higher cost
        cost = np.ones(len(df)) * 1.0  # Base $1 per 100g
        
        # Adjust based on nutrition grade
        if 'nutrition_grade_fr' in df.columns:
            grade_multiplier = {'a': 1.3, 'b': 1.1, 'c': 1.0, 'd': 0.8, 'e': 0.7}
            cost *= df['nutrition_grade_fr'].map(grade_multiplier).fillna(1.0)
        
        # Adjust based on protein content
        cost *= (1 + df['proteins_100g'].fillna(0) / 100)
        
        # Add some randomness for realism
        np.random.seed(42)
        cost *= (0.8 + 0.4 * np.random.rand(len(df)))
        
        return cost
    
    def optimize_meal(self, 
                     target_calories=2000,
                     min_protein=50,
                     max_sugar=50,
                     min_fiber=25,
                     max_sodium=2300,
                     max_cost=20,
                     food_pool=None):
        """
        Optimize a meal plan using linear programming.
        
        Parameters:
        -----------
        target_calories : float
            Target calories (will allow ±10%)
        min_protein : float
            Minimum protein in grams
        max_sugar : float
            Maximum sugar in grams
        min_fiber : float
            Minimum fiber in grams
        max_sodium : float
            Maximum sodium in mg
        max_cost : float
            Maximum cost in dollars
        food_pool : DataFrame
            Custom food pool to use
        
        Returns:
        --------
        DataFrame with selected foods and quantities
        """
        if food_pool is None:
            if self.food_pool is None:
                raise ValueError("No food pool available. Run create_food_pool() first.")
            food_pool = self.food_pool
        
        print("\n" + "="*80)
        print("OPTIMIZING MEAL PLAN")
        print("="*80)
        print(f"\nTargets:")
        print(f"  Calories: {target_calories} kcal (±10%)")
        print(f"  Protein: ≥{min_protein}g")
        print(f"  Sugar: ≤{max_sugar}g")
        print(f"  Fiber: ≥{min_fiber}g")
        print(f"  Sodium: ≤{max_sodium}mg")
        print(f"  Cost: ≤${max_cost:.2f}")
        print(f"\nOptimizing from {len(food_pool):,} foods...")
        
        # Create optimization problem
        prob = LpProblem("Meal_Optimizer", LpMinimize)
        
        # Decision variables: quantity of each food in 100g units
        food_vars = {}
        for idx in food_pool.index:
            food_vars[idx] = LpVariable(f"food_{idx}", lowBound=0, upBound=20)  # Max 2kg per food
        
        # Objective: Minimize cost
        prob += lpSum([
            food_vars[idx] * food_pool.loc[idx, 'cost_per_100g']
            for idx in food_pool.index
        ]), "Total_Cost"
        
        # Constraint 1: Calories (±10%)
        calories = lpSum([
            food_vars[idx] * food_pool.loc[idx, 'energy_100g']
            for idx in food_pool.index
        ])
        prob += calories >= target_calories * 0.9, "Min_Calories"
        prob += calories <= target_calories * 1.1, "Max_Calories"
        
        # Constraint 2: Minimum protein
        protein = lpSum([
            food_vars[idx] * food_pool.loc[idx, 'proteins_100g']
            for idx in food_pool.index
        ])
        prob += protein >= min_protein, "Min_Protein"
        
        # Constraint 3: Maximum sugar
        sugar = lpSum([
            food_vars[idx] * food_pool.loc[idx, 'sugars_100g']
            for idx in food_pool.index
        ])
        prob += sugar <= max_sugar, "Max_Sugar"
        
        # Constraint 4: Minimum fiber (if data available)
        fiber_values = food_pool['fiber_100g'].fillna(0)
        if fiber_values.sum() > 0:
            fiber = lpSum([
                food_vars[idx] * fiber_values.loc[idx]
                for idx in food_pool.index
            ])
            prob += fiber >= min_fiber, "Min_Fiber"
        
        # Constraint 5: Maximum sodium
        sodium_values = food_pool['sodium_100g'].fillna(0) * 1000  # Convert g to mg
        sodium = lpSum([
            food_vars[idx] * sodium_values.loc[idx]
            for idx in food_pool.index
        ])
        prob += sodium <= max_sodium, "Max_Sodium"
        
        # Constraint 6: Food variety (no single food > 40% of total)
        total_quantity = lpSum([food_vars[idx] for idx in food_pool.index])
        for idx in food_pool.index:
            prob += food_vars[idx] <= 0.4 * total_quantity, f"Max_Proportion_{idx}"
        
        # Constraint 7: Minimum total quantity (at least 500g of food)
        prob += total_quantity >= 5, "Min_Total_Quantity"
        
        # Solve the problem
        print("\nSolving optimization problem...")
        prob.solve()
        
        # Check status
        status = LpStatus[prob.status]
        print(f"Status: {status}")
        
        if status != 'Optimal':
            print("⚠️  Could not find optimal solution. Try relaxing constraints.")
            return None
        
        # Extract results
        selected_foods = []
        total_cost = 0
        total_calories = 0
        total_protein = 0
        total_sugar = 0
        total_fiber = 0
        total_sodium = 0
        
        for idx in food_pool.index:
            quantity = value(food_vars[idx])
            if quantity > 0.01:  # Include if > 1g
                food_info = food_pool.loc[idx]
                
                calories_contrib = quantity * food_info['energy_100g']
                protein_contrib = quantity * food_info['proteins_100g']
                sugar_contrib = quantity * food_info['sugars_100g']
                fiber_contrib = quantity * food_info.get('fiber_100g', 0)
                sodium_contrib = quantity * food_info.get('sodium_100g', 0) * 1000
                cost_contrib = quantity * food_info['cost_per_100g']
                
                selected_foods.append({
                    'Food': food_info['product_name'],
                    'Nutri-Score': food_info.get('nutrition_grade_fr', 'N/A'),
                    'Quantity (g)': round(quantity * 100, 1),
                    'Calories (kcal)': round(calories_contrib, 1),
                    'Protein (g)': round(protein_contrib, 1),
                    'Sugar (g)': round(sugar_contrib, 1),
                    'Fiber (g)': round(fiber_contrib, 1),
                    'Sodium (mg)': round(sodium_contrib, 0),
                    'Cost ($)': round(cost_contrib, 2)
                })
                
                total_cost += cost_contrib
                total_calories += calories_contrib
                total_protein += protein_contrib
                total_sugar += sugar_contrib
                total_fiber += fiber_contrib
                total_sodium += sodium_contrib
        
        # Create results dataframe
        results_df = pd.DataFrame(selected_foods)
        
        # Sort by quantity
        results_df = results_df.sort_values('Quantity (g)', ascending=False)
        
        # Print results
        print("\n" + "="*80)
        print("OPTIMIZED MEAL PLAN")
        print("="*80)
        print(results_df.to_string(index=False))
        
        print("\n" + "-"*80)
        print("TOTALS:")
        print("-"*80)
        print(f"  Total Cost:    ${total_cost:.2f}")
        print(f"  Total Calories: {total_calories:.0f} kcal (target: {target_calories})")
        print(f"  Total Protein:  {total_protein:.1f}g (target: ≥{min_protein}g)")
        print(f"  Total Sugar:    {total_sugar:.1f}g (target: ≤{max_sugar}g)")
        print(f"  Total Fiber:    {total_fiber:.1f}g (target: ≥{min_fiber}g)")
        print(f"  Total Sodium:   {total_sodium:.0f}mg (target: ≤{max_sodium}mg)")
        
        # Check if constraints are met
        print("\n" + "-"*80)
        print("CONSTRAINT CHECK:")
        print("-"*80)
        constraints_met = True
        
        if target_calories * 0.9 <= total_calories <= target_calories * 1.1:
            print("  ✓ Calories within target range")
        else:
            print("  ✗ Calories outside target range")
            constraints_met = False
        
        if total_protein >= min_protein:
            print("  ✓ Protein target met")
        else:
            print("  ✗ Protein below target")
            constraints_met = False
        
        if total_sugar <= max_sugar:
            print("  ✓ Sugar below maximum")
        else:
            print("  ✗ Sugar exceeds maximum")
            constraints_met = False
        
        if total_fiber >= min_fiber:
            print("  ✓ Fiber target met")
        else:
            print("  ⚠️  Fiber below target (data may be incomplete)")
        
        if total_sodium <= max_sodium:
            print("  ✓ Sodium below maximum")
        else:
            print("  ✗ Sodium exceeds maximum")
            constraints_met = False
        
        if total_cost <= max_cost:
            print("  ✓ Cost within budget")
        else:
            print("  ✗ Cost exceeds budget")
            constraints_met = False
        
        if constraints_met:
            print("\n✅ All major constraints satisfied!")
        else:
            print("\n⚠️  Some constraints not satisfied")
        
        return results_df
    
    def generate_shopping_list(self, meal_plan):
        """Generate a shopping list from a meal plan."""
        if meal_plan is None or len(meal_plan) == 0:
            return None
        
        print("\n" + "="*80)
        print("SHOPPING LIST")
        print("="*80)
        
        for idx, row in meal_plan.iterrows():
            print(f"  [ ] {row['Food']}: {row['Quantity (g)']}g (${row['Cost ($)']:.2f})")
        
        print(f"\nTotal Cost: ${meal_plan['Cost ($)'].sum():.2f}")
        
        return meal_plan[['Food', 'Quantity (g)', 'Cost ($)']]
    
    def optimize_weekly_plan(self, days=7, variety=True, **daily_targets):
        """
        Optimize a multi-day meal plan.
        
        Parameters:
        -----------
        days : int
            Number of days to plan
        variety : bool
            Ensure different foods each day
        **daily_targets : dict
            Daily nutrition targets (same as optimize_meal)
        """
        print("\n" + "="*80)
        print(f"OPTIMIZING {days}-DAY MEAL PLAN")
        print("="*80)
        
        weekly_plans = []
        used_foods = set()
        
        for day in range(1, days + 1):
            print(f"\n{'='*80}")
            print(f"DAY {day}")
            print(f"{'='*80}")
            
            # Create food pool excluding recently used foods for variety
            if variety and len(used_foods) > 0:
                available_pool = self.food_pool[
                    ~self.food_pool['product_name'].isin(used_foods)
                ].copy()
                
                if len(available_pool) < 50:
                    # Reset if pool gets too small
                    used_foods = set()
                    available_pool = self.food_pool.copy()
            else:
                available_pool = self.food_pool.copy()
            
            # Optimize for this day
            daily_plan = self.optimize_meal(food_pool=available_pool, **daily_targets)
            
            if daily_plan is not None:
                daily_plan['Day'] = day
                weekly_plans.append(daily_plan)
                
                # Add used foods to set
                used_foods.update(daily_plan['Food'].tolist())
            else:
                print(f"⚠️  Could not optimize day {day}")
        
        if len(weekly_plans) == 0:
            return None
        
        # Combine all days
        weekly_df = pd.concat(weekly_plans, ignore_index=True)
        
        # Summary
        print("\n" + "="*80)
        print(f"{days}-DAY MEAL PLAN SUMMARY")
        print("="*80)
        print(f"\nTotal Foods: {len(weekly_df)}")
        print(f"Unique Foods: {weekly_df['Food'].nunique()}")
        print(f"Total Cost: ${weekly_df['Cost ($)'].sum():.2f}")
        print(f"Avg Daily Cost: ${weekly_df['Cost ($)'].sum() / days:.2f}")
        print(f"\nTotal Calories: {weekly_df['Calories (kcal)'].sum():.0f} kcal")
        print(f"Avg Daily Calories: {weekly_df['Calories (kcal)'].sum() / days:.0f} kcal")
        
        return weekly_df


def demo():
    """Run demonstration of meal optimization."""
    print("="*80)
    print("MEAL OPTIMIZATION DEMO")
    print("="*80)
    
    # Check if PuLP is installed
    if not PULP_AVAILABLE:
        print("\n❌ PuLP is not installed!")
        print("\nTo use meal optimization, install PuLP:")
        print("  pip install pulp")
        return
    
    # Initialize optimizer
    optimizer = MealOptimizer(processed_dir='data/processed')
    
    # Load data
    print("\nStep 1: Loading data...")
    optimizer.load_data(sample_size=5000)  # Use 5K for demo
    
    # Create food pool (healthy foods only)
    print("\nStep 2: Creating food pool...")
    optimizer.create_food_pool(
        filters={'nutrition_grade_fr': ['a', 'b', 'c']},
        max_foods=300
    )
    
    # Example 1: Standard meal plan
    print("\n" + "#"*80)
    print("EXAMPLE 1: Standard Daily Meal Plan")
    print("#"*80)
    
    meal_plan = optimizer.optimize_meal(
        target_calories=2000,
        min_protein=60,
        max_sugar=50,
        min_fiber=30,
        max_sodium=2300,
        max_cost=15
    )
    
    if meal_plan is not None:
        optimizer.generate_shopping_list(meal_plan)
    
    # Example 2: High-protein, low-carb
    print("\n" + "#"*80)
    print("EXAMPLE 2: High-Protein, Low-Sugar Meal Plan")
    print("#"*80)
    
    meal_plan_hp = optimizer.optimize_meal(
        target_calories=2000,
        min_protein=100,  # High protein
        max_sugar=30,     # Low sugar
        min_fiber=25,
        max_sodium=2000,
        max_cost=20
    )
    
    # Example 3: Budget-friendly
    print("\n" + "#"*80)
    print("EXAMPLE 3: Budget-Friendly Meal Plan")
    print("#"*80)
    
    meal_plan_budget = optimizer.optimize_meal(
        target_calories=1800,
        min_protein=50,
        max_sugar=60,
        min_fiber=20,
        max_sodium=2500,
        max_cost=8  # Low budget
    )
    
    print("\n" + "="*80)
    print("✅ DEMO COMPLETE!")
    print("="*80)
    print("\n💡 To use in your own code:")
    print("```python")
    print("from meal_optimizer import MealOptimizer")
    print("optimizer = MealOptimizer()")
    print("optimizer.load_data()")
    print("optimizer.create_food_pool()")
    print("plan = optimizer.optimize_meal(target_calories=2000, min_protein=60)")
    print("```")


if __name__ == '__main__':
    demo()

