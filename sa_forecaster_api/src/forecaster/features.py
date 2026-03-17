import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Optional
from sklearn.preprocessing import LabelEncoder
import joblib
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Transforms Silver-level CPI data into Gold-level features for ML model.
    Includes Lagged variables, window statistics, and cyclical temporal features.
    """
    def __init__(self, lag_steps: int = 15):
        self.lag_steps = lag_steps
        base_path = Path(__file__).resolve().parent.parent.parent
        self.silver_data_path = base_path / "data" / "silver" / "CPI_silver.csv"
        self.gold_data_path = base_path / "data" / "gold" / "CPI_gold.csv"
        self.model_dir = base_path / "models"

    def _create_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates auto-regressive lags within each category group."""
        df = df.sort_values(['Category', 'Date']).copy()
        
        for i in range(1, self.lag_steps + 1):
            df[f'Value_{i}'] = df.groupby('Category')['Value'].shift(i)
        
        return df

    def _calculate_vectorized_trend(self, lag_data: pd.DataFrame) -> np.ndarray:
        """
        Calculates the linear slope (trend) across a horizontal set of lags.
        Math: (sum(xi - x_mean)(yi - y_mean)) / sum(xi - x_mean)^2
        """
        n = lag_data.shape[1]
        if n < 2:
            return np.zeros(len(lag_data))
        
        x = np.arange(n)
        x_mean = x.mean()
        
        # Centering x and calculating denominator once
        x_centered = x - x_mean
        denominator = (x_centered**2).sum()
        
        # Centering y (lag_data) and calculating numerator
        y_mean = lag_data.mean(axis=1)
        y_centered = lag_data.subtract(y_mean, axis=0)
        
        numerator = (y_centered.multiply(x_centered, axis=1)).sum(axis=1)
        return numerator / denominator
    
    def get_encoder(self, gold_df, cat_cols: List[str]) -> dict:
        """Get label encoder for each categorical variable."""
        enc_dict = {}
        for col in cat_cols:
            le = LabelEncoder()
            enc_dict[col] = le.fit(gold_df[col])

        return enc_dict

    def ts_stats_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates rolling window statistics (Mean, Std, Trend) based on lags.
        This is a public method as it's also called during live inference.
        """
        lag_cols = [f'Value_{i}' for i in range(1, self.lag_steps + 1)]
        lag_data = df[lag_cols]
        diff_data = lag_data.diff(axis=1).iloc[:,1:4]
        diff_data = diff_data.rename(columns={col:f'diff_{col}' for col in diff_data.columns})

        # Summary Stats
        df[f'past_{self.lag_steps}_mean'] = lag_data.mean(axis=1)
        df[f'past_{self.lag_steps}_std'] = lag_data.std(axis=1)
        df[f'past_{self.lag_steps}_max'] = lag_data.max(axis=1)
        df[f'past_{self.lag_steps}_min'] = lag_data.min(axis=1)
        df[f'past_{self.lag_steps}_median'] = lag_data.median(axis=1)
        df[f'past_{self.lag_steps}_pct_median'] = lag_data.pct_change(axis=1).median(axis=1)
        df[f'past_{self.lag_steps}_pct_std'] = lag_data.pct_change(axis=1).std(axis=1)
        
        # Trend Stats (Long-term vs Short-term)
        df['trend_long'] = self._calculate_vectorized_trend(lag_data)
        df['trend_short'] = self._calculate_vectorized_trend(
            lag_data.iloc[:, :max(3, self.lag_steps // 2)])
        
        df = pd.concat([df,diff_data],axis=1)
        return df

    def add_cyclical_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encodes Month as Sine/Cosine to preserve circular distance."""
        df['month'] = df['Date'].dt.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        return df
    
    def add_interaction_features(self, df: pd.DataFrame,) -> pd.DataFrame:
        """Encodes Month as Sine/Cosine to preserve circular distance."""
        df['category_month'] = df['month'].astype(str) + "_" + df['Category'].astype('str')

        return df

    def transform(self, df: Optional[pd.DataFrame] = None, is_inference: bool = False) -> pd.DataFrame:
        """
        The main pipeline. 
        If is_inference is True, it expects a single row per category and doesn't drop NaNs.
        """
        if df is None:
            df = pd.read_csv(self.silver_data_path)
            df['Date'] = pd.to_datetime(df['Date'])

        # 1. Create Lags
        df = self._create_lags(df)
        
        # 2. Drop rows that don't have enough history (only for training)
        if not is_inference:
            df = df.dropna().reset_index(drop=True)
        
        # 3. Add TS Stats
        df = self.ts_stats_features(df)
        
        # 4. Add Cyclical Features
        df = self.add_cyclical_time_features(df)

        # 5. Add Interaction Features
        df = self.add_interaction_features(df)
        
        logger.info("Feature engineering complete. Shape: %s", df.shape)
        return df

    def get_feature_list(self) -> List[str]:
        """Returns the list of columns the encoder should actually use."""
        # This excludes 'Category', 'Date', and the target 'Value'
        dummy_df = pd.DataFrame(columns=[f'Value_{i}' for i in range(1, self.lag_steps + 1)])
        # Trigger stats to get names
        dummy_df = self.ts_stats_features(dummy_df)
        
        lag_cols = [f'Value_{i}' for i in range(1, self.lag_steps + 1)]
        stat_cols = [c for c in dummy_df.columns if c not in lag_cols]
        time_cols = ['month', 'month_sin', 'month_cos']
        interaction_cols = ['category_month']
        
        return ['Category'] + lag_cols + stat_cols + time_cols + interaction_cols

    def save_gold_resources(self, df: pd.DataFrame,cat_cols):
        self.gold_data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.gold_data_path, index=False)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        encoders = self.get_encoder(df,cat_cols)
        encoder_filename = f"cpi_encoder_{timestamp}.joblib"
        save_path = self.model_dir / encoder_filename
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(encoders, save_path)
        
        # Update "latest" symlink/pointer
        joblib.dump(encoders, self.model_dir / "CPI_encoder_latest.joblib")
        
        logger.info("Gold data saved to %s. Encoders saved to %s", self.gold_data_path,
                    self.model_dir / "CPI_encoder_latest.joblib" )

if __name__ == "__main__":
    fe = FeatureEngineer(lag_steps=15)
    gold_df = fe.transform()
    fe.save_gold_resources(gold_df,cat_cols=['month','category_month','Category'])
    
    logger.info("Features used for encodering: %s", fe.get_feature_list())
