import dill
import pickle
import os
import sys 
from src.exception import CustomException
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
import optuna
from optuna.trial import Trial
from src.logger import logging



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok= True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_path)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, X_test, y_train, y_test, models):
    report = {}
    for i in models.items():
        model_name = i[0]
        Model = i[1][1]
        model = i[1][0]
        logging.info(f'{model}')

        def objective(trial):
             
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
            
                # 'XGBoost' : {
                # 'max_depth': trial.suggest_int('max_depth', 1, 9),
                # 'learning_rate': trial.suggest_float('learning_rate', 0.001, 1.0),
                # 'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
                # 'min_child_weight': trial.suggest_int('min_child_weight', 1, 13),
                # 'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log = True),
                # 'subsample': trial.suggest_float('subsample', 0.01, 1.0, log = True),
                # 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
                # 'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log = True),
                # 'eval_metric': 'rmse'
                # }

             }
            
        

            params = parameters[model_name]
            ML = model.fit(X_train, y_train)
            predictions =  ML.predict(X_test)
            error = np.sqrt(mean_squared_error(y_test, predictions))
                                              
            return error

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=5)
        best_parameters = study.best_params

        final_model = Model(**best_parameters)
        final_model = final_model.fit(X_train,y_train)
        outsample_predictions = final_model.predict(X_test)
        test_error  = np.sqrt(mean_squared_error(y_test, outsample_predictions))
        report[model_name] = test_error

    return report
