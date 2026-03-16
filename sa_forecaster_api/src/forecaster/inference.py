import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import joblib
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CPIPredictor:
    def __init__(self, model_path: Optional[Path] = None, data_path: Optional[Path] = None,
                 encoder_path: Optional[Path] = None):
        # Configuration - Can be overridden by env variables in production
        base_path = Path(__file__).resolve().parent.parent.parent
        self.model_path = model_path or base_path / "models" / "CPI_model_latest.joblib"
        self.encoder_path = encoder_path or base_path / "models" / "CPI_encoder_latest.joblib"
        self.gold_data_path = data_path or base_path / "data" / "gold" / "CPI_gold.csv"
        self.predictions_output_path = base_path / "data" / "predictions"
        
        # Internal state
        self.model = None
        self.sel_features = None

    def load_resources(self):
        """Loads model and metadata."""
        resource_dict = joblib.load(self.model_path)
        self.sel_features = resource_dict['features']
        self.model = resource_dict['model']
        logger.info("Model and features loaded from %s", self.model_path)

    def load_gold_resources(self) -> pd.DataFrame:
        if not self.gold_data_path.exists():
            raise FileNotFoundError(f"Gold data not found at {self.gold_data_path}")
        df = pd.read_csv(self.gold_data_path)
        df['Date'] = pd.to_datetime(df['Date'])

        #get encoder
        encoder_dict = joblib.load(self.encoder_path)

        return df,encoder_dict

    def prepare_next_month_features(self, df: pd.DataFrame, ts_stats_fn,
                                    cyclical_time_fn, interaction_fn) -> pd.DataFrame:
        """
        Takes the current dataset and prepares the feature row for the immediate next month.
        """
        latest_date = df['Date'].max()
        latest_data = df[df['Date'] == latest_date].copy()
        
        # Identify 'Value_i' columns
        lag_cols = [col for col in df.columns if col.startswith('Value_')]
        
        # Re-align Lags: This month's 'Value' becomes Next month's 'Value_1'
        # Drop the oldest lag (Value_15) and shift others
        new_features = latest_data[['Category', 'Date', 'Value'] + lag_cols[:-1]].copy()
        
        # Rename columns to standard Value_1...Value_15
        new_col_names = ['Category', 'Date'] + [f'Value_{i}' for i in range(1, len(lag_cols) + 1)]
        new_features.columns = new_col_names

        # Advance timeline
        new_features['Date'] = new_features['Date'] + pd.DateOffset(months=1)

        # Apply domain-specific feature engineering (stats, etc.)
        return interaction_fn(cyclical_time_fn(ts_stats_fn(new_features)))

    def run_forecast_pipeline(self, feature_engineer_stats_fn, cyclical_time_fn,
                               interaction_fn, steps: int = None):
        """
        Production entry point: Handles the full recursive forecasting loop.
        """
        self.load_resources()
        gold_data, encoder_dict = self.load_gold_resources()
        last_actual_date = gold_data['Date'].max()

        # If steps not provided, calculate based on current date
        if steps is None:
            steps = (datetime.now().year - last_actual_date.year) * 12 + (datetime.now().month - last_actual_date.month)
            steps = max(1, steps)

        logger.info("Starting recursive forecast for %d steps from %s", steps, last_actual_date.strftime('%Y-%m'))

        all_new_predictions = []
        current_working_df = gold_data.copy()

        for step in range(1, steps + 1):
            # 1. Prepare features for the next step
            inference_row = self.prepare_next_month_features(current_working_df,
                                                              feature_engineer_stats_fn,
                                                              cyclical_time_fn,
                                                              interaction_fn)
            
            # 2. Predict
            to_model = inference_row.copy()
            for col in encoder_dict:
                if col in to_model.columns:
                    to_model[col] = encoder_dict[col].transform(to_model[col])

            inference_row['Value'] = self.model.predict(to_model[self.sel_features])

            
            # 3. Store and Update working data for next iteration
            current_working_df = pd.concat([current_working_df, inference_row], ignore_index=True)

            
            all_new_predictions.append(inference_row)
            
            logger.info("Step %d/%d completed for date: %s",
                         step, steps, inference_row['Date'].max().strftime('%Y-%m'))

        
        final_forecasts = pd.concat(all_new_predictions, ignore_index=True)

        # round predictions to 2 decimal places for storage consistency
        final_forecasts['Value'] = final_forecasts['Value'].round(2)

        self.save_predictions(final_forecasts, steps)
        return final_forecasts

    def save_predictions(self, df: pd.DataFrame, steps: int):
        '''Saves the forecast results to a CSV file.'''
        month_str = datetime.now().strftime("%Y-%m")
        output_dir = self.predictions_output_path / month_str
        output_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = output_dir / f"forecast_results_{steps}m.csv"
        df.to_csv(file_path, index=False)
        logger.info("Final predictions saved to %s", file_path)

if __name__ == "__main__":
    from sa_forecaster_api.src.forecaster.features import FeatureEngineer
    
    fe = FeatureEngineer()
    predictor = CPIPredictor()
    
    results = predictor.run_forecast_pipeline(
        feature_engineer_stats_fn=fe.ts_stats_features,
        cyclical_time_fn=fe.add_cyclical_time_features,
        interaction_fn=fe.add_interaction_features
    )
    print(results[['Category', 'Date', 'Value']].head())