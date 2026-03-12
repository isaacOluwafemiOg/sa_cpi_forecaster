import logging
import json
import joblib
import optuna
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from catboost import Pool, CatBoostRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CPITrainer:
    def __init__(self, target_col: str = 'Value'):
        self.target_col = target_col
        base_path = Path(__file__).resolve().parent.parent.parent
        self.gold_data_path = base_path / "data" / "gold" / "CPI_gold.csv"
        self.model_dir = base_path / "models"
        self.metrics_path = base_path / "models" / "model_metrics.json"

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.gold_data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        # Sort by date for TimeSeriesSplit
        df = df.sort_values('Date').reset_index(drop=True)
        return df

    def get_features_and_target(self, df: pd.DataFrame, feature_cols: list) -> tuple:
        X = df[feature_cols]
        y = df[self.target_col]
        return X, y

    def run_feature_selection(self, df: pd.DataFrame, initial_features: list) -> list:
        """
        Streamlined feature selection using CatBoost importance.
        In production, we often do this once and keep the features stable.
        """
        logger.info("Starting feature importance evaluation...")
        X = df[initial_features]
        y = df[self.target_col]
        
        model = CatBoostRegressor(iterations=500, silent=True, random_seed=42)
        model.fit(X, y)
        
        importances = pd.Series(model.get_feature_importance(), index=initial_features)
        # Keep features that contribute to the model
        selected = importances[importances > 0].index.tolist()
        logger.info("Selected %d features out of %d", len(selected), len(initial_features))
        return selected

    def objective(self, trial, X, y):
        """Optuna objective for CatBoost hyperparams."""
        param = {
            "iterations": trial.suggest_int("iterations", 200, 1000),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 1e-9, 10.0, log=True),
            "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"]),
            "eval_metric": "RMSE",
            "loss_function": "RMSE",
            "random_seed": 42,
            "silent": True
        }

        # TimeSeries Cross-Validation
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []

        for train_idx, val_idx in tscv.split(X):
            X_t, X_v = X.iloc[train_idx], X.iloc[val_idx]
            y_t, y_v = y.iloc[train_idx], y.iloc[val_idx]
            
            model = CatBoostRegressor(**param)
            model.fit(X_t, y_t, eval_set=(X_v, y_v), early_stopping_rounds=50)
            preds = model.predict(X_v)
            scores.append(root_mean_squared_error(y_v, preds))

        return np.mean(scores)

    def train_and_save(self, feature_list: list):
        """Full training pipeline: Tune -> Fit Final -> Save."""
        df = self.load_data()
        X, y = self.get_features_and_target(df, feature_list)

        logger.info("Starting hyperparameter tuning...")
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: self.objective(trial, X, y), n_trials=20)
        
        best_params = study.best_params
        logger.info("Best Params: %s", best_params)

        #store best RMSE as a proxy for model health
        best_rmse = study.best_value

        # Final Fit on ALL data
        final_model = CatBoostRegressor(**best_params, silent=True)
        final_model.fit(X, y)


        self._save_artifacts(final_model, feature_list, best_params, best_rmse)
        return best_rmse

    def _save_artifacts(self, model, features, params, score):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        # Save Model + Feature List
        model_filename = f"cpi_model_{timestamp}.joblib"
        save_path = self.model_dir / model_filename
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({"model": model, "features": features}, save_path)
        
        # Update "latest" symlink/pointer
        joblib.dump({"model": model, "features": features}, self.model_dir / "CPI_model_latest.joblib")

        # Save Metrics JSON for Dashboard
        metrics = {
            "last_train_date": timestamp,
            "rmse": float(score),
            "features_used": features,
            "hyperparameters": params
        }
        with open(self.metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Model saved to {save_path} with RMSE: {score}")

if __name__ == "__main__":
    from sa_forecaster_api.src.forecaster.features import FeatureEngineer
    
    # 1. Get feature names from FeatureEngineer
    fe = FeatureEngineer()

    feature_names = fe.get_feature_list()

    # 2. Train
    trainer = CPITrainer()
    trainer.train_and_save(feature_names)