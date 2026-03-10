import pandas as pd
from pathlib import Path

class DataCleaner:
    '''
    Cleans and processes CPI data for analysis.
    '''
    def __init__(self,target_categories=None):
        self.target_categories = target_categories if target_categories is not None else [
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
        'Restaurants and accommodation services'
    ]
        self.raw_data_path = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "CPI_latest.xlsx"

    def load_to_dataframe(self) -> pd.DataFrame:
        """Search for the Excel file in the data folder and load it."""
        latest_file = self.raw_data_path
        
        # Stats SA Excel files often have multiple header rows; adjustment might be needed
        df = pd.read_excel(latest_file)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
        

    # ==========================================
    # 3. Data Filtering and Cleaning
    # ==========================================

    def filter_cpi_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters the raw CPI dataframe for the specific categories and urban areas.
        Renames columns for better interpretability.

        Args:
            df (pd.DataFrame): The raw dataframe from Stats SA.

        Returns:
            pd.DataFrame: A filtered and renamed dataframe.
        """
        # Filtering based on specific Stats SA codes:
        # H04: Category name
        # H13: Geographical area (All urban areas is the national benchmark)
        df_filtered = df[
            (df['H04'].isin(self.target_categories)) & (df['H13'] == 'All urban areas')
            ].copy()

        # Mapping internal codes to human-readable names for cleaner code downstream
        rename_map = {
            'H04': 'Category',
        }
        
        # Selecting and renaming specific columns (Adjust these based on your specific Excel structure)
        df_filtered = df_filtered.rename(columns=rename_map)

        return df_filtered
    


    # ==========================================
    # 4. Refining Category Granularity
    # ==========================================

    def refine_education_category(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters out specific sub-categories within the 'Education' group to 
        ensure we are only using the primary 'Education' index.
        
        This prevents double-counting or using inconsistent sub-indices.

        Args:
            df (pd.DataFrame): The filtered CPI dataframe.

        Returns:
            pd.DataFrame: Dataframe with unwanted education sub-categories removed.
        """
        # first ensure there are more than one unique instances of 'Education' in the Category column
        if df[df['Category'] == 'Education'].shape[0] <= 1:
            print("No multiple sub-categories found for 'Education'.")
            
            #Dropping extra features that aren't needed
            df_refined = df.drop(['H01','H02','H03','H05','H06','H13',
                                  'H17','H18','H24','H25'],axis=1)
            return df_refined
        
        # List of sub-categories to remove as identified in the H05 column
        # These represent granular components that might deviate from the main index
        unwanted_edu_subcats = [
            'University boarding fees',
            'Tertiary education',
            'Primary and secondary education',
            'Education including boarding fees'
        ]
        
        df_refined = df[~(df['H05'].isin(unwanted_edu_subcats))].copy()
        
        initial_count = len(df)
        final_count = len(df_refined)
        
        print(f"Refinement complete: Removed {initial_count - final_count} sub-category rows.")

        #Dropping extra features that aren't needed
        df_refined = df_refined.drop(['H01','H02','H03','H05','H06','H13',
                                  'H17','H18','H24','H25'],axis=1)
        
        return df_refined
    
    # ==========================================
    # 5. Time-Series Structural Alignment
    # ==========================================

    def format_cpi_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates a standardized date range and assigns it as column headers.
        
        Args:
            df (pd.DataFrame): The filtered CPI dataframe.

        Returns:
            pd.DataFrame: Dataframe with formatted date columns.
        """
        # extract start mmyyyy and end mmyyyy from dataframe columns such that df.columns[1]
        # is the start date in the format 'MOmmyyyy' and df.columns[-1] is the end date in the same format
        start_date_str = df.columns[1]
        end_date_str = df.columns[-1]
        start_date = pd.to_datetime(start_date_str, format='MO%m%Y')
        end_date = pd.to_datetime(end_date_str, format='MO%m%Y')
        
        # Generate a sequence of dates from start to end with monthly frequency
        date_index = pd.date_range(start=start_date, end=end_date, freq='MS')

        
        formatted_dates = [f"{d.month}-{d.year}" for d in date_index]

        # Create the new column list
        new_columns = ['Category'] + formatted_dates

        print(f"Generated {len(formatted_dates)} date columns from {start_date_str} to {end_date_str}.")
        print(f"Expected columns: {new_columns[:5]} ... {new_columns[-5:]}")  # Show a sample of the new columns
        
        # Validation: Ensure column count matches data shape
        if len(new_columns) != df.shape[1]:
            print(f"Warning: Column mismatch. Data has {df.shape[1]} columns, but we generated {len(new_columns)}.")
        
        df.columns = new_columns
        return df
    
    # ==========================================
    # 6. Reshaping to Long Format
    # ==========================================

    def reshape_cpi_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pivots the dataframe from wide (months as columns) to long (months as rows).
        Converts dates to datetime objects and sorts the data chronologically.

        Args:
            df (pd.DataFrame): The wide-format CPI dataframe.

        Returns:
            pd.DataFrame: A cleaned, long-format dataframe sorted by Category and Date.
        """
        # 1. Melt the data: 'Category' stays, all date columns become rows in 'Date'
        df_long = pd.melt(df, id_vars='Category', var_name='Date', value_name='Value')

        # 2. Convert 'Date' column to datetime objects
        # Our format was 'Month-Year' (e.g., '1-2008')
        df_long['Date'] = pd.to_datetime(df_long['Date'], format='%m-%Y')

        # 3. Ensure 'Value' is numeric (important for modeling)
        df_long['Value'] = pd.to_numeric(df_long['Value'], errors='coerce')

        # 4. Sort the data
        # Sorting by Category then Date ensures a clean sequence for feature engineering
        df_long = df_long.sort_values(by=['Category', 'Date']).reset_index(drop=True)

        return df_long

    def save_cleaned_data(self, df: pd.DataFrame,file_name: str = "CPI_cleaned.csv") -> None:
        """
        Saves the cleaned dataframe to a specified path in CSV format.

        Args:
            df (pd.DataFrame): The cleaned long-format dataframe.
            output_path (str): The file path where the cleaned data should be saved.
        """
        output_dir = Path(__file__).resolve().parent.parent.parent / "data" / "bronze"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / file_name
        
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved successfully to {output_file}")


if __name__ == "__main__":
    cleaner = DataCleaner()

    data = cleaner.load_to_dataframe()
    
    # Apply the filter
    cpi_filtered = cleaner.filter_cpi_data(data)

    # Validate results
    print(f"Categories selected: {len(cpi_filtered['Category'].unique())} of {len(cleaner.target_categories)}")
    print(cpi_filtered['Category'].unique())

    edu_filtered = cleaner.refine_education_category(cpi_filtered)

    # Execute column renaming and date formatting
    edu_filtered = cleaner.format_cpi_columns(edu_filtered)

    # Apply the transformation
    cpi_long = cleaner.reshape_cpi_data(edu_filtered)

    # Preview the results
    print(f"Long format shape: {cpi_long.shape}")

    # Save the cleaned data    
    cleaner.save_cleaned_data(cpi_long)



        