from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import optuna
from catboost import Pool, CatBoostRegressor
from sklearn.metrics import root_mean_squared_error
from datetime import datetime
CAT_COLS = []

class Trainer:
    '''
    Handles the training of forecasting models for CPI data.
    '''
    def __init__(self):
        self.gold_data_path = Path(__file__).resolve().parent.parent.parent / "data" / "gold" / "CPI_final.csv"


    def load_to_dataframe(self):
        """Load csv file from the data folder."""
        df = pd.read_csv(self.gold_data_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df

    def data_split(self, df, train_size, date_col='Date'):
        """
        Splits the dataframe into features and predictors for 
        training and validation sets based on a specified date column.

        Args:
            df (pd.DataFrame): The input dataframe to split.
            train_size (float): The proportion of the data to include in the training set (between 0 and 1).
            date_col (str): The name of the date column to use for splitting.

        Returns:
            pd.DataFrame,pd.Series,pd.DataFrame,pd.Series: The training and validation features and targets.
        """
        # Ensure the date column is in datetime format
        df[date_col] = pd.to_datetime(df[date_col])

        split_date = df[date_col].quantile(train_size)

        # Split the dataframe into training and validation sets
        train_df = df[df[date_col] < split_date].reset_index(drop=True)
        valid_df = df[df[date_col] >= split_date].reset_index(drop=True)

        print(f"Data split at date: {split_date}. Training set: {train_df.shape}, Validation set: {valid_df.shape}")

        # separate features and target
        x_train = train_df.drop(['Value', date_col,'Category'], axis=1)
        Y_train = train_df['Value']
        x_valid = valid_df.drop(['Value', date_col,'Category'], axis=1)
        Y_valid = valid_df['Value']
        
        return x_train, Y_train, x_valid, Y_valid

    
    # ==========================================
    # 1. Feature Selection (Catboost's built-in feature importance)
    # ==========================================

    def reg_objective_unk(self, Xtrain, ytrain, Xtest, ytest, no_remove,
                          rem, steps, iterations, cat_cols):
        '''
        Performs feature selection using CatBoost's built-in feature importance.
        Iteratively trains a CatBoost model, evaluates feature importance, 
        and eliminates the least important features
        '''
        obj = 'RMSE'
        
        cols_to_eliminate = {}
        step_lls = {}
        for step in range(steps):
            col_oof_ll = []
            oof_unimp = []
            if step==0:
                npyc_df = Xtrain
            else:
                drop_cols = cols_to_eliminate[f'step_{step-1}']
                npyc_df = Xtrain.drop(drop_cols,axis=1)

            step_level_imps = {k:[] for k in npyc_df.columns}
            for Xs, ys in [
                [(Xtrain[npyc_df.columns],Xtest[npyc_df.columns]),
                (ytrain.copy(),ytest.copy())]
            ]:

                X_train, X_test = Xs
                y_train, y_test = ys

                
                #display(y_train.value_counts())
                params = {'iterations': iterations,'learning_rate': 0.05,'depth': 5,
                        "loss_function":obj,
                        'eval_metric': obj,
                        'random_seed': 42,'verbose': int(iterations/2),
                        'task_type':'CPU','use_best_model':True}
                
                
                
                prepped_X =X_train 
                prepped_Xtest =X_test
                

                cat_features = [i for i, col in enumerate(X_train.columns) if col in cat_cols]
                
                train_pool = Pool(prepped_X, y_train,cat_features=cat_features)
                test_pool = Pool(prepped_Xtest, y_test, cat_features=cat_features)
                
                model = CatBoostRegressor(**params)

                model.fit(train_pool,eval_set=test_pool)

                feature_importances = model.get_feature_importance()
                

                for i,col in enumerate(X_train.columns):
                    step_level_imps[col].append(feature_importances[i])

                
                pred = model.predict(prepped_Xtest)
                
                ind_cv = root_mean_squared_error(y_test, pred)
                
                col_oof_ll.append(ind_cv)
                
            print(f"rmse step {step}:  {np.mean(col_oof_ll)}")
            step_lls[step] = np.mean(col_oof_ll)

            step_imps_avg = {k:np.mean(v) for k,v in step_level_imps.items()}
            importance_df = pd.DataFrame({
                'Feature': step_imps_avg.keys(),
                'Importance': step_imps_avg.values()
                }).sort_values(by='Importance', ascending=False)

            importance_df = importance_df[~(importance_df['Feature'].isin(no_remove))]
            oof_unimp = importance_df[importance_df['Importance']<=0]['Feature'].to_list()
            min_rem_size = rem
            if len(oof_unimp)<min_rem_size:
                
                oof_unimp = importance_df.tail(min_rem_size)['Feature'].to_list()

            print('to remove: ',oof_unimp)
            if step==0:
                cols_to_eliminate['step_0'] = oof_unimp
            elif step<steps-1:
                cols_to_eliminate[f'step_{step}'] = cols_to_eliminate[f'step_{step-1}'] + oof_unimp
                
        min_u = min(step_lls.values())
        chosen_step = [x for x in step_lls.keys() if step_lls[x]==min_u]
        chosen_step = chosen_step[-1]
        if chosen_step>0:
            chosen_features = [col for col in Xtrain.columns if col not in cols_to_eliminate[f'step_{chosen_step-1}']]
        else:
            chosen_features = Xtrain.columns.tolist()
        print(f'chosen step is {chosen_step} with value: {min_u}')
        return chosen_features, step_lls, cols_to_eliminate

    def reg_objective(self, trial, X_train, y_train, X_test, y_test,cat_cols):
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

        
        cat_features = [i for i, col in enumerate(X_train.columns) if col in cat_cols]
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        test_pool = Pool(X_test, y_test, cat_features=cat_features)

        
        model = CatBoostRegressor(
            iterations=iterations, learning_rate=learning_rate, depth=depth, l2_leaf_reg=l2_leaf_reg,
            random_strength=random_strength, grow_policy=grow_policy, max_leaves=max_leaves,
            # Bootstrap params
            bootstrap_type=bootstrap_type, subsample=subsample, bagging_temperature=bagging_temperature,
            colsample_bylevel=colsample_bylevel, min_data_in_leaf=min_data_in_leaf,
            
            objective='RMSE', task_type='CPU', random_state=42, silent=True,
            use_best_model=True, eval_metric='RMSE'
        )

        model.fit(train_pool, eval_set=test_pool)
        
        y_pred = model.predict(X_test)
        score = root_mean_squared_error(y_test, y_pred)
        
        return score
    
    def tune_hyperparameters(self, Xtrain, ytrain, Xvalid, yvalid,
                              sel_fe,trials, timeout,cat_cols):
        rstudy = optuna.create_study(direction='minimize')
        rstudy.optimize(lambda trial: self.reg_objective(trial,Xtrain[sel_fe],ytrain,
                                                         Xvalid[sel_fe],yvalid,cat_cols),
                                                         gc_after_trial=True,n_trials=trials,
                                                           timeout=timeout)
        
        return rstudy.best_params
    
    def fit_model(self, hyperparams, X_train, y_train, X_test, y_test,
                  cat_cols,sel_fe):
        '''fit model on all of the available data'''
        
        #combine predictors and features of train and test into one set of predictors and features
        X = pd.concat([X_train,X_test]).reset_index(drop=True)
        y = pd.concat([y_train,y_test]).reset_index(drop=True)

        obj = 'RMSE'
        params = {'objective': obj,
                'random_seed': 42,'silent': True,
                'task_type':'CPU',**hyperparams}

        prepped_X =X[sel_feats] 

        cat_features = [i for i, col in enumerate(sel_fe) if col in cat_cols]

        train_pool = Pool(prepped_X, y,cat_features=cat_features)

        model = CatBoostRegressor(**params)
        model.fit(train_pool)

        return model
    
    def save_model(self, model,sel_fe, model_name: str = "CPI_model.joblib",is_latest=True) -> None:
        """
        Saves the fitted model to disk using joblib.
        
        Args:
            model: The fitted model object.
            save_path (Path): The directory path where models should be saved.
        """
        date = datetime.now().strftime("%Y%m")
        output_dir = Path(__file__).resolve().parent.parent.parent / "models" / date
        
        output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        output_file = output_dir / model_name
        
        joblib.dump((sel_fe,model), output_file)

        if is_latest:
            output_latest = Path(__file__).resolve().parent.parent.parent / "models"
            joblib.dump((sel_fe,model), output_latest / "CPI_model_latest.joblib") 

        print(f"Model saved to {output_file}")

if __name__ == "__main__":
    trainer = Trainer()
    loaded_data = trainer.load_to_dataframe()
    
    # Perform data split
    X_train, y_train, X_valid, y_valid = trainer.data_split(loaded_data, train_size=0.8)
    
    sel_feats, lls,cols = trainer.reg_objective_unk(X_train,y_train,
                                                    X_valid,y_valid,[],
                                                    1,X_train.shape[1]-5,100,
                                                    cat_cols=CAT_COLS)
    
    hyp = trainer.tune_hyperparameters(X_train, y_train, X_valid, y_valid, sel_feats,trials=20,
                                       cat_cols=CAT_COLS, timeout=3600)

    cpi_model = trainer.fit_model(hyp, X_train, y_train, X_valid, y_valid, CAT_COLS, sel_feats)

    trainer.save_model(cpi_model, sel_feats)