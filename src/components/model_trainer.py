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

        # X_train = pd.DataFrame(X_train)
        # X_test = pd.DataFrame(X_test)
        # y_train = pd.DataFrame(y_train)
        # y_test = pd.DataFrame(y_test)

        logging.info(f'X shape : {X_train.shape}')
        models = {
              'CatBoost' : CatBoostRegressor()
            #'RandomForest' : RandomForestRegressor(),
            #'AdaBoost' : AdaBoostRegressor(),
           #  'XGBoost' : XGBRegressor(),
           # 'LGBM' : LGBMRegressor()
        }
        model = CatBoostRegressor()
        logging.info('Model is Training')
        def objective(trial):
            errors  = []
            parameters = {
                'loss_function' : 'RMSE',
                'learning_rate' : trial.suggest_loguniform('learning_rate', 0.01, 0.1),
                'max_depth': trial.suggest_categorical('max_depth', [5,7,9,11,13,15]),
               'random_state': trial.suggest_categorical('random_state', [27]),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 300),
                'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 2.0, 10.0),
                'max_bin': trial.suggest_int('max_bin', 200, 400),
                'n_estimators':  trial.suggest_int('n_estimators',  10, 2500)
            }
            
            # fold = StratifiedKFold(n_splits = 5)
            # for train_index, test_index in fold.split(X_train, y_train):
            #     print(train_index)
            #     X_train_cv, X_test_cv = X_train[0:44,train_index] , X_test[test_index]
            #     y_train_cv, y_test_cv = y_train[0:44,train_index] , y_test[test_index]

            #     model = CatBoostRegressor(**parameters)

            # model.fit(X_train,y_train,eval_set=[(X_train_cv,y_train_cv),(X_test_cv, y_test_cv)], early_stopping_rounds=10, verbose=False)
            eval_pool = Pool(X_test,y_test)
            model = CatBoostRegressor(**parameters)
            model.fit(X_train, y_train, eval_set= eval_pool, early_stopping_rounds= 20)
            predictions = model.predict(X_test)
            error = np.sqrt(mean_squared_error(y_test,predictions))
            # errors.append(error)

            return error
        

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)
        
        params = study.best_params
        final_model = CatBoostRegressor(**params)
        final_model = final_model.fit(X_train, y_train)
        final_predictions = final_model.predict(X_test)
        final_error = np.sqrt(mean_squared_error(y_test , final_predictions))
        # model = RandomForestRegressor()
        #logging.info('{}'.format(X_test.shape))
        logging.info('Model Training Done')

        #outsample_predictions = model.predict(X_train)

        #error = np.sqrt(mean_squared_error(y_train, outsample_predictions))

        #model_report = evaluate_models(X_train, X_test, y_train, y_test, models, parameters)
                     
        return final_error
