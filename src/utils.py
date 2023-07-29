import dill
import pickle
import os
import sys 
from src.exception import CustomException
import numpy as np
from sklearn.metrics import mean_squared_error
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
    try:
        report = {}
        for i in len(list(models)):
            model = list(models.values())[i]
            model_name  = list(models.key())[i]

            Model = model.fit(X_train, y_train)
            logging.info('Done traininng {} model'.format(model_name))

            #Making predcitions

            insample_predictions = model.predict(X_train)
            outsample_predictions = model.predict(X_test)


            #Computing Errors
            train_error = np.sqrt(mean_squared_error(y_train, insample_predictions))
            test_error  = np.sqrt(mean_squared_error(y_test, outsample_predictions))
            error_list = [train_error, test_error]

            logging.info('Done computing errors')


            report[model_name] = error_list

            return report
    except Exception as e:
        raise CustomException(e,sys)