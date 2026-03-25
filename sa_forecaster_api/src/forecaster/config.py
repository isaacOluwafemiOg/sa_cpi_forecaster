from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List

class Settings(BaseSettings):
    # INFERENCE
    CHOICE_MODEL: str = Field(default = 'CPI_model_latest',
                              description="Model to use for inference")
    # MODEL TRAINING
    HYPERPARAM_OPTIM_ITER: int = 100

    #FEATURE ENGINEERING
    LAG_COUNT: int = 15
    CAT_COLS: List[str] = Field(default = ['month','category_month','Category'],
                                description='features to treat as categories in the modelling process')

    #INGESTION
    STATSSA_BASE_URL: str = "https://www.statssa.gov.za/timeseriesdata/Excel/"
    PUB_CODE: str = Field(default = 'P0141', description="publication code for CPI on the statssa site")

    
    # Configuration for loading files
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore" # Ignore extra variables in .env not defined here
    )

# Instantiate to load variables immediately
settings = Settings()
