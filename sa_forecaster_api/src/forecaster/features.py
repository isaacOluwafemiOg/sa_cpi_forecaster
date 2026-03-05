import pandas as pd
from pathlib import Path

class FeatureEngineer:
    def __init__(self):
        pass

    def load_to_dataframe(self, bronze_data_path, file_pattern="*.csv"):
        """Search for the csv file in the data folder and load it."""
        folder = Path(bronze_data_path)
        files = list(folder.glob(file_pattern))
        if not files:
            raise FileNotFoundError(f"No {file_pattern} found in {folder}")
        try:
            df = pd.read_csv(files[0])
            print(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"An error occurred while loading the data: {e}")
            return pd.DataFrame()
        
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
        data = df.copy()
        
        # We group by 'Category' to ensure shifts happen within the same CPI class.
        # We use .shift(-i) because the data is sorted with the newest dates at the top.
        for i in range(1, steps + 1):
            data[f'Value_{i}'] = data.groupby('Category')['Value'].shift(-i)

        # Drop rows with NaN values created by the shift (the oldest 'steps' months)
        data = data.dropna(axis=0)
        
        # Reset index for a clean slate before modeling
        data = data.reset_index(drop=True)
        
        print(f"Feature engineering complete. Created {steps} lag features.")
        print(f"New shape: {data.shape}")
        
        return data


if __name__ == "__main__":
    featureMaker = FeatureEngineer()
    data = featureMaker.load_to_dataframe("../data/bronze/P0141")

    # Generating 12 months of lags (1 year of historical context for each prediction)
    LAG_STEPS = 15
    cpi_with_lags = featureMaker.create_lagged_features(data, steps=LAG_STEPS)

    # Preview the head to see Value_1, Value_2... alongside the target 'Value'
    cpi_with_lags.head()