from pathlib import Path
from datetime import datetime
import logging
import json
import joblib
import optuna
import numpy as np
import pandas as pd
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

    def load_gold_resources(self) -> pd.DataFrame:
        '''Loads the Gold-level CPI resources for training.'''
        df = pd.read_csv(self.gold_data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        # Sort by date for TimeSeriesSplit
        df = df.sort_values('Date').reset_index(drop=True)

        #get encoder
        encoder_dict = joblib.load(self.model_dir / "CPI_encoder_latest.joblib")
        return df,encoder_dict

    def get_features_and_target(self, df: pd.DataFrame, feature_cols: list) -> tuple:
        '''Extracts feature matrix X and target vector y from the DataFrame.'''
        X = df[feature_cols]
        y = df[self.target_col]
        return X, y

    def run_feature_selection(self, df: pd.DataFrame, initial_features: list,
                              cat_cols) -> list:
        """
        Streamlined feature selection using CatBoost importance.
        """
        logger.info("Starting feature importance evaluation...")
        X = df[initial_features]
        y = df[self.target_col]
        
        cat_features = [i for i, col in enumerate(X.columns) if col in cat_cols]
        train_pool = Pool(X, y, cat_features=cat_features)
        model = CatBoostRegressor(iterations=500, silent=True, random_seed=42)
        model.fit(train_pool)
        
        importances = pd.Series(model.get_feature_importance(), index=initial_features)
        # Keep features that contribute to the model
        selected = importances[importances > importances.quantile(0.25)].index.tolist()
        logger.info("Selected %d features out of %d", len(selected), len(initial_features))
        return selected

    def objective(self, trial, X, y,cat_cols):
        """Optuna objective for CatBoost hyperparams."""
        # --- 1. Basic Tree Params ---
        iterations = trial.suggest_int("iterations", 100, 1000)
        learning_rate = trial.suggest_float("learning_rate", 0.001, 0.3, log=True)
        depth = trial.suggest_int("depth", 4, 10)
        l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True)
        
        # --- 2. Randomness & Noise (New) ---
        random_strength = trial.suggest_float("random_strength", 1e-9, 10.0, log=True)
        
        # --- 3. Tree Growth Strategy (New) ---
        grow_policy = trial.suggest_categorical("grow_policy", ["SymmetricTree",
                                                                 "Depthwise", "Lossguide"])
        
        # Min data in leaf interacts differently depending on grow_policy, 
        # but generally safe to tune.
        min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 1, 100) 

        # If Lossguide, we need to limit the number of leaves specifically
        if grow_policy == "Lossguide":
            max_leaves = trial.suggest_int("max_leaves", 16, 64)
        else:
            max_leaves = None # Ignored for SymmetricTree/Depthwise


        # --- 5. Bootstrapping (Fixed Logic) ---
        bootstrap_type = trial.suggest_categorical("bootstrap_type", ['Bayesian', 'Bernoulli', 'MVS'])
        
        subsample = None
        bagging_temperature = None
        
        if bootstrap_type in ['Bernoulli', 'MVS']:
            subsample = trial.suggest_float("subsample", 0.5, 1.0)
        elif bootstrap_type == 'Bayesian':
            bagging_temperature = trial.suggest_float("bagging_temperature", 0.0, 10.0)

        # --- 6. Feature Sampling ---
        colsample_bylevel = trial.suggest_float("colsample_bylevel", 0.4, 1.0)


        # TimeSeries Cross-Validation
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []

        for train_idx, val_idx in tscv.split(X):
            X_t, X_v = X.iloc[train_idx], X.iloc[val_idx]
            y_t, y_v = y.iloc[train_idx], y.iloc[val_idx]

            cat_features = [i for i, col in enumerate(X_t.columns) if col in cat_cols]
            train_pool = Pool(X_t, y_t, cat_features=cat_features)
            test_pool = Pool(X_v, y_v, cat_features=cat_features)
            
            model = CatBoostRegressor(
                iterations=iterations, learning_rate=learning_rate,
                  depth=depth, l2_leaf_reg=l2_leaf_reg,
                  random_strength=random_strength, grow_policy=grow_policy,
                  max_leaves=max_leaves,
                # Bootstrap params
                bootstrap_type=bootstrap_type, subsample=subsample, 
                bagging_temperature=bagging_temperature,
                colsample_bylevel=colsample_bylevel, min_data_in_leaf=min_data_in_leaf,
                
                objective='RMSE', task_type='CPU', random_state=42, silent=True,
                use_best_model=True, eval_metric='RMSE'
                )
            
            model.fit(train_pool, eval_set=test_pool)
            preds = model.predict(X_v)
            scores.append(root_mean_squared_error(y_v, preds))

        return np.mean(scores)

    def train_and_save(self, feature_list: list):
        """Full training pipeline: Tune -> Fit Final -> Save."""
        df,encoder_dict = self.load_gold_resources()

        df = df.copy()
        for col in encoder_dict:
            df[col] = encoder_dict[col].transform(df[col])

        feature_list = self.run_feature_selection(df,feature_list,
                                                  cat_cols=encoder_dict.keys())

        X, y = self.get_features_and_target(df, feature_list)
        
        logger.info("Starting hyperparameter tuning...")
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: self.objective(trial, X, y, cat_cols=encoder_dict.keys()),
                        n_trials=100)
        
        best_params = study.best_params
        logger.info("Best Params: %s", best_params)

        #store best RMSE as a proxy for model health
        best_rmse = study.best_value

        # Final Fit on ALL data
        final_model = CatBoostRegressor(**best_params, silent=True,
                                        objective='RMSE', task_type='CPU', 
                                        random_state=42)
        final_model.fit(X, y)

        importances = pd.Series(final_model.get_feature_importance(), index=X.columns).to_dict()


        self._save_artifacts(final_model, importances,feature_list, best_params, best_rmse)
        return best_rmse

    def _save_artifacts(self, model, importances,feature_list, params, score):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        # Save Model + Feature List
        model_filename = f"cpi_model_{timestamp}.joblib"
        save_path = self.model_dir / model_filename
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({"model": model, "features": feature_list}, save_path)
        
        # Update "latest" symlink/pointer
        joblib.dump({"model": model, "features": feature_list},
                     self.model_dir / "CPI_model_latest.joblib")

        # Save Metrics JSON for Dashboard
        metrics = {
            "last_train_date": timestamp,
            "rmse": float(score),
            "features_importance": importances,
            "hyperparameters": params
        }
        
        with open(self.metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        
        logger.info("Model saved to %s with RMSE:%.4f", save_path, score)

if __name__ == "__main__":
    from sa_forecaster_api.src.forecaster.features import FeatureEngineer
    
    # 1. Get feature names from FeatureEngineer
    fe = FeatureEngineer()

    feature_names = fe.get_feature_list()

    # 2. Train
    trainer = CPITrainer()
    trainer.train_and_save(feature_names)
