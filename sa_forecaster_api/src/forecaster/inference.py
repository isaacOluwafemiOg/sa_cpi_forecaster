import pandas as pd
from pathlib import Path

from typing import Any

from sa_forecaster_api.src.forecaster.features import FeatureEngineer
import joblib


class CPIPredictor:
    def __init__(self):
        self.model_path = Path(__file__).resolve().parent.parent.parent / "models" / "cpi_models_latest.pkl"

    def load_models(self) -> dict:
        """
        Loads pre-trained models from specified paths.

        Args:
            model_paths (dict): A dictionary mapping category names to their model file paths.

        Returns:
            dict: A dictionary of loaded model objects.
        """
        try:
            models = joblib.load(self.model_path)
            print(f"Models loaded successfully from {self.model_path}")
            return models
        except Exception as e:
            print(f"An error occurred while loading the models: {e}")
            return {}

    # ==========================================
    # 11. Inference Preparation
    # ==========================================

    def prepare_inference_data(self, df: pd.DataFrame, 
                               pre_target_month: str = '7-2023') -> pd.DataFrame:
        """
        Extracts the most recent feature row to be used for nowcasting the specified month.
        
        Args:
            df (pd.DataFrame): The dataframe containing lagged features.
            target_month (str): The month immediately preceding the target month.

            For example, to predict August 2023, we use the row for July 2023 as the source of features.

        Returns:
            pd.DataFrame: A dataframe containing the features for the specified month forecast.
        """
        # 1. Filter for the month we want to predict 
        # We will shift these to become the features for the target month.
        latest_data = df[df['Date'] == pre_target_month].copy()
        
        # 2. Re-aligning Lags for the future month:
        # To predict August:
        # August_Lag_1 = July_Value
        # August_Lag_2 = July_Lag_1 ... and so on.
        
        # Identify all 'Value_i' columns present in the training data
        lag_cols = [col for col in df.columns if col.startswith('Value_')]
        
        # Create the new feature set for inference by shifting the lags down by one month
        # For instance if predicting for August, the 'Value' from July becomes 'Value_1' for August.
        inference_row = latest_data[['Category', 'Value'] + lag_cols[:-1]].copy()
        
        # Rename columns to match the model's expected input (Value_1, Value_2, etc.)
        new_col_names = ['Category'] + [f'Value_{i}' for i in range(1, len(lag_cols) + 1)]
        inference_row.columns = new_col_names
        
        print(f"Inference features prepared for {len(inference_row)} categories.")
        return inference_row
    
    # ==========================================
    # 12. Final Prediction & Submission Preparation
    # ==========================================

    def generate_nowcast(self, row: pd.Series, models: dict) -> float:
        """
        Retrieves the correct model for a category and generates a CPI prediction.

        Args:
            row (pd.Series): A single row of features including 'Category'.
            models (dict): The dictionary of fitted model objects.

        Returns:
            float: The predicted CPI value for August 2023.
        """
        category = row['Category']
        
        # Retrieve the model specific to this category
        model = models.get(category)
        
        if model is None:
            raise ValueError(f"No model found for category: {category}")

        # Prepare features: Drop the category label and reshape for scikit-learn (1, n_features)
        # .values.reshape(1, -1) is a clean way to handle single-row inference
        features = row.drop('Category').values.reshape(1, -1)
        
        # Generate prediction
        prediction = model.predict(features)
        
        # Return as a scalar float
        return float(prediction[0])
    )

    
if __name__ == "__main__":
    # Example usage of the prepare_inference_data function
    inferrer = CPIPredictor()

    models = inferrer.load_models()

    featureMaker = FeatureEngineer()
    loaded_data = featureMaker.load_to_dataframe()
    cpi_with_lags = featureMaker.create_lagged_features(loaded_data, steps=15)

    inference_features = inferrer.prepare_inference_data(cpi_with_lags, pre_target_month='7-2023')

    # Display features for a quick sanity check
    inference_features.head()

    inference_features['Value'] = inference_features.apply(
            lambda x: inferrer.generate_nowcast(x, models), axis=1
        )

