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
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from src.utils import evaluate_models
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    model_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_path = ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting into X_train, X_test, y_train, y_test')

            X_train = train_array[:,:-1]
            y_train = train_array[:,-1]
            X_test = test_array[:,:-1]
            y_test = test_array[:,-1]

            models = {
                'RandomForest' : RandomForestRegressor(),
                'AdaBoost' : AdaBoostRegressor(),
                'CatBoost' : CatBoostRegressor(),
                'XGBoost' : XGBRegressor(),
                'LGBM' : LGBMRegressor()
            }

            model_report = evaluate_models(X_train, X_test, y_train, y_test, models)

            logging.info('Model Training Done')
                                    
            return model_report
        except Exception as e:
            raise CustomException(e ,sys)