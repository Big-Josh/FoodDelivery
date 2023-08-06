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


    def initiate_model_trainer(self, input_train_array, input_test_array, output_train_array, output_test_array):
            
        logging.info('Splitting into X_train, X_test, y_train, y_test')

        X_train = input_train_array
        X_test = input_test_array
        y_train = output_train_array.ravel()
        y_test = output_test_array.ravel()

        models = {
                 'LGBM' : [LGBMRegressor(), LGBMRegressor],
                'CatBoost' : [CatBoostRegressor(), CatBoostRegressor]
               # 'XGBoost' : XGBRegressor(),
               
        }
    
        logging.info('Training Models')

        model_report = evaluate_models(X_train, X_test, y_train, y_test, models)

        return model_report


