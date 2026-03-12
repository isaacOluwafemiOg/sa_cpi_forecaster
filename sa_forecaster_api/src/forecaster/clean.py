import pandas as pd
import logging
import re
from pathlib import Path
from typing import List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataCleaner:
    """
    Cleans and processes raw Stats SA CPI Excel data into a standardized long format.
    """
    def __init__(self, target_categories: Optional[List[str]] = None):
        self.target_categories = target_categories or [
            'CPI Headline',
            'Food and non alcoholic beverages',
            'Alcoholic beverages and tobacco',
            'Clothing and footwear',
            'Housing and utilities',
            'Health',
            'Transport',
            'Information and communication',
            'Recreation, sport and culture',
            'Education',
            'Restaurants and accommodation services',
        ]
        
        base_path = Path(__file__).resolve().parent.parent.parent
        self.raw_data_path = base_path / "data" / "raw" / "CPI_latest.xlsx"
        self.output_dir = base_path / "data" / "silver"

    def load_raw_data(self) -> pd.DataFrame:
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {self.raw_data_path}")
        
        # Stats SA files sometimes have metadata in early rows; 
        # but read_excel handles most standard COICOP sheets well.
        df = pd.read_excel(self.raw_data_path)
        logger.info("Loaded raw data with shape: %s", df.shape)
        return df

    def filter_and_rename(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filters for Urban Areas and target COICOP categories."""
        # H04: Product Category, H13: Geographic Area
        mask = (df['H04'].isin(self.target_categories)) & (df['H13'] == 'All urban areas')
        df_filtered = df[mask].copy()

        # Check for Education sub-categories (H05 contains sub-labels)
        # We only want the primary category, so we filter out rows where H05 is specific
        unwanted_edu_subcats = [
            'University boarding fees', 'Tertiary education', 
            'Primary and secondary education', 'Education including boarding fees'
        ]
        df_filtered = df_filtered[~df_filtered['H05'].isin(unwanted_edu_subcats)]

        # Keep only Category column and the actual data columns (starting with 'MO')
        date_cols = [col for col in df.columns if str(col).startswith('MO')]
        df_filtered = df_filtered[['H04'] + date_cols]
        df_filtered = df_filtered.rename(columns={'H04': 'Category'})
        
        logger.info("Filtered data to %d categories.", len(df_filtered))
        return df_filtered

    def standardize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms column names from MOmmYYYY to standard Date format."""
        new_columns = {'Category': 'Category'}
        
        for col in df.columns:
            if col == 'Category': continue
            
            # Extract month and year from 'MO012008' pattern
            match = re.search(r'MO(\d{2})(\d{4})', str(col))
            if match:
                month, year = match.groups()
                # Create a standardized date string (YYYY-MM-01)
                new_columns[col] = f"{year}-{month}-01"
            else:
                logger.warning("Column %s did not match date pattern and was ignored.", col)

        df = df.rename(columns=new_columns)
        return df

    def melt_to_long_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pivots the table and ensures data types are correct for ML."""
        df_long = pd.melt(df, id_vars='Category', var_name='Date', value_name='Value')
        
        # Ensure correct types
        df_long['Date'] = pd.to_datetime(df_long['Date'])
        df_long['Value'] = pd.to_numeric(df_long['Value'], errors='coerce')
        
        # Drop rows with missing values (often at the end of the time series)
        initial_len = len(df_long)
        df_long = df_long.dropna(subset=['Value'])
        
        if len(df_long) < initial_len:
            logger.info("Dropped %d rows with null values.", initial_len - len(df_long))

        df_long = df_long.sort_values(['Category', 'Date']).reset_index(drop=True)
        return df_long

    def process_pipeline(self):
        """Executes the full cleaning pipeline."""
        df = self.load_raw_data()
        df = self.filter_and_rename(df)
        df = self.standardize_dates(df)
        df = self.melt_to_long_format(df)
        
        # Final Validation: Do we have all categories?
        missing = set(self.target_categories) - set(df['Category'].unique())
        if missing:
            logger.warning("The following target categories were not found: %s", missing)
            
        self.save_data(df)
        return df

    def save_data(self, df: pd.DataFrame):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "CPI_silver.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaner.process_pipeline()