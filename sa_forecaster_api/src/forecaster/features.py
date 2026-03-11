import pandas as pd
from pathlib import Path
import numpy as np

class FeatureEngineer:
    '''
    Creates features for CPI data to be used in forecasting models.
    '''
    def __init__(self):
        self.bronze_data_path = Path(__file__).resolve().parent.parent.parent / "data" / "bronze" / "CPI_cleaned.csv"

    def load_to_dataframe(self):
        """Search for the csv file in the data folder and load it."""

        df = pd.read_csv(self.bronze_data_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
        
    # ==========================================
    # 1. Lagged Observations
    # ==========================================

    def create_lagged_features(self, df: pd.DataFrame, steps: int) -> pd.DataFrame:
        """
        Creates auto-regressive features by shifting the CPI values.
        
        This function treats each CPI category as an independent time series 
        to prevent data leakage between different categories.

        Args:
            df (pd.DataFrame): Long-format dataframe (Expected order: Descending by Date).
            steps (int): The number of previous months to use as features.

        Returns:
            pd.DataFrame: Dataframe with new columns 'Value_1', 'Value_2', etc., 
                        representing past observations.
        """
        data = df.sort_values(by=['Category', 'Date'],
                               ascending=[True, True]).copy()  # Ensure correct order for shifting
    
        # We group by 'Category' to ensure shifts happen within the same CPI class.
        # We use .shift(i) because the data is sorted with the oldest dates at the top.
        for i in range(1, steps + 1):
            data[f'Value_{i}'] = data.groupby('Category')['Value'].shift(i)

        # Drop rows with NaN values created by the shift (the oldest 'steps' months)
        data = data.dropna(axis=0)
        
        # Reset index for a clean slate before modeling
        data = data.reset_index(drop=True)
        
        print(f"Feature engineering complete. Created {steps} lag features.")
        print(f"New shape: {data.shape}")
        
        return data
    
    # ==========================================
    # 2. Feature Engineering (Time Series Summary Statistics)
    # ==========================================

    def vectorized_trend(self, lag_data):
        """Calculates the linear trend slope across the lags."""
        n = lag_data.shape[1]
        if n < 2:
            return np.zeros(len(lag_data))
        x = np.arange(n)
        x_mean = x.mean()
        # (sum(xi - xmean)(yi - ymean)) / sum(xi - xmean)^2
        numerator = ((lag_data.subtract(lag_data.mean(axis=1), axis=0))
                    .multiply(x - x_mean, axis=1)).sum(axis=1)
        denominator = ((x - x_mean)**2).sum()
        return numerator / denominator

    def ts_stats_features(self, df: pd.DataFrame, steps: int) -> pd.DataFrame:
        """
        Creates auto-regressive features by shifting the CPI values.
        
        This function treats each CPI category as an independent time series 
        to prevent data leakage between different categories.

        Args:
            df (pd.DataFrame): Long-format dataframe (Expected order: Descending by Date).
            steps (int): The number of previous months to use as features.

        Returns:
            pd.DataFrame: Dataframe with new columns 'Value_1', 'Value_2', etc., 
                        representing past observations.
        """
        lagged_data = df[[f'Value_{i}' for i in range(1, steps + 1)]]
        
        df_dict = {}
        df_dict[f'past_{steps}_mean'] = lagged_data.mean(axis=1)
        df_dict[f'past_{steps}_std'] = lagged_data.std(axis=1)
        df_dict[f'past_{steps}_min'] = lagged_data.min(axis=1)
        df_dict[f'past_{steps}_max'] = lagged_data.max(axis=1)
        df_dict[f'past_{steps}_median'] = lagged_data.median(axis=1)
        df_dict[f'past_{steps}_skew'] = lagged_data.skew(axis=1)
        df_dict[f'past_{steps}_kurt'] = lagged_data.kurtosis(axis=1)
        df_dict[f'past_{steps}_trend'] = self.vectorized_trend(lagged_data)
        df_dict[f'past_{int(steps/2)}_trend'] = self.vectorized_trend(lagged_data.iloc[:, :int(steps/2)])
        
        df_stats = pd.DataFrame(df_dict)

        # Concatenate the original dataframe with the new statistics dataframe
        data = pd.concat([df, df_stats], axis=1)
        
        print("Time series summary statistics generated.")
        print(f"New shape: {data.shape}")
        
        return data

    
    
    def save_featured_data(self, df: pd.DataFrame, file_name: str = "CPI_final.csv") -> None:
        """
        Saves the dataframe with engineered features to a specified path in CSV format.

        Args:
            df (pd.DataFrame): The dataframe with lagged features.
        """
        output_dir = Path(__file__).resolve().parent.parent.parent / "data" / "gold"
        output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        output_file = output_dir / file_name
        
        df.to_csv(output_file, index=False)
        print(f"Data with engineered features saved successfully to {output_file}")
        

if __name__ == "__main__":
    featureMaker = FeatureEngineer()
    loaded_data = featureMaker.load_to_dataframe()

    # Generating 15 months of lags
    LAG_STEPS = 15
    cpi_with_lags = featureMaker.create_lagged_features(loaded_data, steps=LAG_STEPS)

    cpi_with_lag_stats = featureMaker.ts_stats_features(cpi_with_lags, steps=LAG_STEPS)

    # Save the featured data
    featureMaker.save_featured_data(cpi_with_lag_stats)
