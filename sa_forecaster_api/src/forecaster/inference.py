from datetime import datetime

import pandas as pd
from pathlib import Path

from typing import Any

from sa_forecaster_api.src.forecaster.features import FeatureEngineer
import joblib


class CPIPredictor:
    def __init__(self):
        self.model_path = Path(__file__).resolve().parent.parent.parent / "models" / "CPI_model_latest.joblib"
        self.gold_data_path = Path(__file__).resolve().parent.parent.parent / "data" / "gold" / "CPI_final.csv"

    def load_model(self) -> dict:
        """
        Loads pre-trained model from specified path.

        Returns:
            loaded model object.
        """
        model = joblib.load(self.model_path)
        print(f"Model loaded successfully from {self.model_path}")
        return model
    
    def load_gold_data(self):
        """Load csv file from the data folder."""
        df = pd.read_csv(self.gold_data_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    
    # ==========================================
    # 11. Inference Preparation (August 2023)
    # ==========================================

    def prepare_inference_data(self, df: pd.DataFrame, steps: int, ts_stats_features) -> pd.DataFrame:
        """
        Extracts the most recent feature row to be used for nowcasting the next month.
        
        Args:
            df (pd.DataFrame): The dataframe containing lagged features.
            steps (int): The number of steps to consider for feature engineering.
            ts_stats_features (function): The function to calculate time series statistics.

        Returns:
            pd.DataFrame: A dataframe containing the features for August 2023 forecast.
        """
        latest_date = df['Date'].max()

        # 1. Filter for the most recent month 
        latest_data = df[df['Date'] == latest_date].copy()
        
        # 2. Re-aligning Lags for the future month:
        
        # Identify all 'Value_i' columns present in the training data
        lag_cols = [col for col in df.columns if col.startswith('Value_')]
        
        # Create the new feature set for inference by shifting the lag columns to represent the next month
        cpi_with_lag = latest_data[['Category','Date', 'Value'] + lag_cols[:-1]].copy()
        
        # Rename columns to match the model's expected input (Value_1, Value_2, etc.)
        new_col_names = ['Category','Date'] + [f'Value_{i}' for i in range(1, len(lag_cols) + 1)]
        cpi_with_lag.columns = new_col_names

        cpi_with_lag['Date'] = cpi_with_lag['Date'] + pd.DateOffset(months=1)

        cpi_with_lag_stats = ts_stats_features(cpi_with_lag, steps=steps)
        
        print(f"Inference features prepared for {len(cpi_with_lag_stats)} categories.")
        return cpi_with_lag_stats
    
    def get_prediction(self, df, model, sel_feats) -> pd.DataFrame:
        '''get prediction for the next month'''
        df['Value'] = model.predict(df[sel_feats])
        
        return df

    

    
    
if __name__ == "__main__":
    # Example usage of the prepare_inference_data function
    inferrer = CPIPredictor()

    curr_date = datetime.now()

    gold_data = inferrer.load_gold_data()

    last_gold_data_date = gold_data['Date'].max()

    #get number of months to forecast
    if isinstance(last_gold_data_date, str):
        last_gold_data_date = datetime.strptime(last_gold_data_date, "%Y-%m-%d")
 
    forecast_steps = (curr_date-last_gold_data_date).days // 30

    print(f"Current date: {curr_date}, Last gold data date: {last_gold_data_date}, Forecast steps: {forecast_steps}")

    sel_feats,model = inferrer.load_model()

    featureMaker = FeatureEngineer()


    for step in range(1, forecast_steps + 1):
        print(f"Preparing inference data for step {step}...")
        inference_features = inferrer.prepare_inference_data(
            gold_data, steps=15, ts_stats_features=featureMaker.ts_stats_features)
        
        prediction_df = inferrer.get_prediction(inference_features,model, sel_feats)
        gold_data = pd.concat([gold_data, prediction_df], ignore_index=True)

        print(f"Prediction for step {step} completed. Latest prediction date: {prediction_df['Date'].max()}")   


