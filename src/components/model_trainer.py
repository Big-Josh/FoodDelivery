import os
import sys
import pandas as np
import numpy as np
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor
)
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor, cv, Pool
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from src.utils import evaluate_models
from sklearn.metrics import mean_squared_error
import optuna
from optuna.trial import Trial
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    model_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_path = ModelTrainerConfig()


    def initiate_model_trainer(self, input_train_array, input_test_array, output_train_array, output_test_array, trial = None):
            
        logging.info('Splitting into X_train, X_test, y_train, y_test')

        X_train = input_train_array
        X_test = input_test_array
        y_train = output_train_array.ravel()
        y_test = output_test_array.ravel()

        models = {
                'CatBoost' : CatBoostRegressor(),
                'RandomForest' : RandomForestRegressor(),
                'XGBoost' : XGBRegressor(),
                'LGBM' : LGBMRegressor()
        }
    
        parameters = {
            'CatBoost' :  {
            'loss_function' : 'RMSE',
            'learning_rate' : trial.suggest_loguniform('learning_rate', 0.01, 0.1),
            'max_depth': trial.suggest_categorical('max_depth', [5,7,9,11,13,15]),
           'random_state': trial.suggest_categorical('random_state', [27]),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 300),
            'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 2.0, 10.0),
            'max_bin': trial.suggest_int('max_bin', 200, 400),
            'n_estimators':  trial.suggest_int('n_estimators',  10, 2500)
            },

            'LGBM' : {
            "objective": "regression",
            "metric": "rmse",
            "n_estimators": trial.suggest_int('n_estimators', 50, 1000),
            "verbosity": -1,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 2**10),
            "subsample": trial.suggest_float("subsample", 0.05, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
            },

            'XGBoost' : {
            'max_depth': trial.suggest_int('max_depth', 1, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 13),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log = True),
            'subsample': trial.suggest_float('subsample', 0.01, 1.0, log = True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log =  True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log = True),
            'eval_metric': 'rmse'
            }
        }


        model_report = evaluate_models(X_train, X_test, y_train, y_test, models, parameters)

        return model_report

