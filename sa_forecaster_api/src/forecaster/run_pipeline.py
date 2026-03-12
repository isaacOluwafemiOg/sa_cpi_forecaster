import logging
import argparse
from pathlib import Path
from datetime import datetime

# Import your modules
from sa_forecaster_api.src.forecaster.ingestion import StatsSAIngestor
from sa_forecaster_api.src.forecaster.clean import DataCleaner
from sa_forecaster_api.src.forecaster.features import FeatureEngineer
from sa_forecaster_api.src.forecaster.train import CPITrainer
from sa_forecaster_api.src.forecaster.inference import CPIPredictor

# Setup centralized logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CPI_Pipeline")

def run_full_pipeline(force_retrain: bool = False):
    """
    Orchestrates the end-to-end flow from data ingestion to inference.
    """
    start_time = datetime.now()
    logger.info("Starting South Africa CPI Nowcasting Pipeline...")

    try:
        # --- 1. DATA INGESTION ---
        logger.info("--- Stage 1: Ingestion ---")
        ingestor = StatsSAIngestor()
        latest_yyyymm = ingestor.find_and_ingest_latest()
        
        if not latest_yyyymm:
            logger.error("Data ingestion failed. Check network or Stats SA website.")
            return

        # --- 2. DATA CLEANING ---
        logger.info("--- Stage 2: Cleaning (Bronze to Silver) ---")
        cleaner = DataCleaner()
        _ = cleaner.process_pipeline()
        
        # --- 3. FEATURE ENGINEERING ---
        logger.info("--- Stage 3: Feature Engineering (Silver to Gold) ---")
        fe = FeatureEngineer(lag_steps=15)
        gold_df = fe.transform()
        fe.save_gold_data(gold_df)
        feature_list = fe.get_feature_list()

        # --- 4. TRAINING (Conditional) ---
        # Logic: We retrain if 'force_retrain' is true OR if it's a new month's data
        # In a real system, you might check a 'last_trained_on' metadata file.
        trainer = CPITrainer()

        if force_retrain:
            logger.info("--- Stage 4: Training (Model Retraining Forced) ---")
            rmse_score = trainer.train_and_save(feature_list)
            logger.info("Retraining complete. New RMSE: %.4f", rmse_score)

        # if there is no model in the models directory, we also need to train
        
        elif not (trainer.model_dir / "CPI_model_latest.joblib").exists():
            logger.info("--- Stage 4: Training (No existing model found, retraining) ---")
            rmse_score = trainer.train_and_save(feature_list)
            logger.info("Initial training complete. RMSE: %.4f", rmse_score)

        else:
            logger.info("--- Stage 4: Training (Skipped, using latest model) ---")

        # --- 5. INFERENCE (Nowcasting) ---
        logger.info("--- Stage 5: Inference (Generating Nowcast) ---")
        predictor = CPIPredictor()
        # We pass the ts_stats and cyclical_time methods to the predictor for sliding-window features
        forecast_results = predictor.run_forecast_pipeline(
            feature_engineer_stats_fn=fe.ts_stats_features,
            cyclical_time_fn=fe.add_cyclical_time_features
        )
        
        # --- 6. SUCCESS & SUMMARY ---
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info("Pipeline completed successfully in %s", duration)
        logger.info("Latest prediction date: %s", forecast_results['Date'].max().strftime('%Y-%m'))

    except Exception as e:
        logger.critical("Pipeline crashed! Error: %s", str(e), exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPI Nowcasting Orchestrator")
    parser.add_argument(
        "--retrain", 
        action="store_true", 
        help="Force a full model retraining and hyperparameter tuning"
    )
    args = parser.parse_args()

    run_full_pipeline(force_retrain=args.retrain)