import dill
import pickle
import os
import sys 
from src.exception import CustomException
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
import optuna
from src.logger import logging



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok= True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_path)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, X_test, y_train, y_test, models, params):
    report = {}
    for i in range(len(list(models))):
        model = list(models.values())[i]
        model_name  = list(models.keys())[i]
        parameter = params[list(models.keys())[i]]

        def objective(study):
            errors = []

            fold = StratifiedKFold(n_splits=5)
        
            for train_index, test_index in fold.split(X_train,y_train):
                X_train_cv, X_test_cv = X_train.iloc[train_index], X_train.iloc[test_index]
                y_train_cv, y_test_cv = y_train.iloc[train_index], y_train.iloc[test_index]

                cv_model = model(**parameter)

                cv_model = cv_model.fit(X_train, y_train)
                prediction = cv_model.predict(X_test)
                error = np.sqrt(mean_squared_error(y_test,prediction))
                errors.append(error)
            return errors
        
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=50)
        best_parameters = study.best_params

        final_model = model(**best_parameters)
        outsample_predictions = final_model.predict(X_test)
        test_error  = np.sqrt(mean_squared_error(y_test, outsample_predictions))
    return test_error

    #     Model = model.fit(X_train, y_train)
    #     logging.info('Done traininng {} model'.format(model_name))

    #     #Making predcitions
    #     insample_predictions = Model.predict(X_train)
    #     outsample_predictions = Model.predict(X_test)

    #     #Computing Errors
    #     train_error = np.sqrt(mean_squared_error(y_train, insample_predictions))
    #     test_error  = np.sqrt(mean_squared_error(y_test, outsample_predictions))

    #     error_list = [train_error, test_error]
    #     logging.info('Done computing errors')

    #     report[model_name] = error_list

    # return report