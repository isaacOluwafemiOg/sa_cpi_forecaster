from sklearn.pipeline import Pipeline
from typing import Any
import pandas as pd
from pathlib import Path
from sklearn.feature_selection import SelectKBest
import joblib

# Scientific computing
from scipy.stats import uniform

# Bayesian Linear Regression model
# ARDRegression is excellent for handling datasets where many features might be irrelevant
from sklearn.linear_model import ARDRegression

# Bayesian Hyperparameter Optimization
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# Set seed for reproducibility if needed later
RANDOM_STATE = 42

class Trainer:
    '''
    Handles the training of forecasting models for CPI data.
    '''
    def __init__(self):
        self.gold_data_path = Path(__file__).resolve().parent.parent.parent / "data" / "gold" / "CPI_final.csv"


    def load_to_dataframe(self):
        """Load csv file from the data folder."""
        df = pd.read_csv(self.gold_data_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df

    # ==========================================
    # 1. Model Fitting Utility
    # ==========================================

    def fit_model_pipeline(self, model: Any, data: pd.DataFrame) -> Any:
        """
        Fits a provided model or pipeline to the given dataset.
        
        Automatically separates the target 'Value' from the features and 
        removes non-numeric metadata columns to prevent training errors.

        Args:
            model (sklearn.pipeline.Pipeline or BaseEstimator): An unfitted model or pipeline.
            data (pd.DataFrame): The training data containing features and the 'Value' target.

        Returns:
            Any: The fitted model/pipeline.
        """
        # Create a copy to prevent modifying the original dataframe
        train_df = data.copy()
        
        # Define features (X) and target (y)
        # We drop 'Value' (target) and metadata that shouldn't be used as features
        X = train_df.drop(columns=['Value', 'Category', 'Date'], errors='ignore')
        y = train_df['Value']
        
        # Fit the model
        model.fit(X, y)
        
        return model
    
    def train_all_categories(self, df: pd.DataFrame, models: dict) -> dict:
        """
        Iterates through each CPI category and fits a unique model for each.
        
        Args:
            df (pd.DataFrame): The dataframe containing lagged features for all categories.
            models (dict): A dictionary mapping category names to their model objects.
            
        Returns:
            dict: A dictionary of fitted model objects.
        """
        fitted_models = {}
        
        print("Starting training for all CPI categories...")
        
        for category_name, model_instance in models.items():
            # Slice the data for the specific category
            category_data = df[df['Category'] == category_name]
            
            if category_data.empty:
                print(f"Warning: No data found for category '{category_name}'. Skipping.")
                continue
                
            # Fit the model using our utility function
            # This function handles dropping 'Category' and 'Date' automatically
            fitted_model = self.fit_model_pipeline(model_instance, category_data)
            
            # Store the fitted model
            fitted_models[category_name] = fitted_model
            print(f"Successfully trained model for: {category_name}")
            
        return fitted_models
    
    def save_models(self, models: dict, model_name: str = "CPI_models.joblib") -> None:
        """
        Saves the fitted models to disk using joblib.
        
        Args:
            models (dict): A dictionary of fitted model objects.
            save_path (Path): The directory path where models should be saved.
        """
        output_dir = Path(__file__).resolve().parent.parent.parent / "data" / "gold"
        output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        output_file = output_dir / model_name
        
        joblib.dump(models, output_file)
        print(f"Models saved to {output_file}")

if __name__ == "__main__":
    trainer = Trainer()
    loaded_data = trainer.load_to_dataframe()
    
    # Example: Define a simple model for each category (replace with actual models)
    category_models = {
        'CPI Headline': ARDRegression(),
        'CPI Food': ARDRegression(),
        'CPI Transport': ARDRegression(),
        'CPI Education': ARDRegression()
    }
    
    fitted_models = trainer.train_all_categories(loaded_data, category_models)

    trainer.save_models(fitted_models)