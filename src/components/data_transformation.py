import sys
import os
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from src.utils import save_object
from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    transformer_path = os.path.join('artifacts','transformer.pkl')


class DataTransformation:
    
    def __init__(self) :
        self.transformer_path = DataTransformationConfig()

    try:
        def transfomer_object(self):

            categorical_columns = [
                'market_id', 'order_protocol', 'cusine'
            ]
        
            numerical_columns = [
            'total_items', 'subtotal', 'num_distinct_items', 'min_item_price',
            'max_item_price', 'total_onshift_dashers', 'total_busy_dashers',
            'total_outstanding_orders','estimated_order_place_duration',
            'estimated_store_to_consumer_driving_duration','hour_of_order','day_of_order'
            ]
        
            categorical_pipeline = Pipeline(
                steps = [('OneHotEncoding', OneHotEncoder())]
            )
        
            numerical_pipeline = Pipeline(
                steps = [('StandardScaling', StandardScaler()),
                        ('PCA', PCA(n_components=4)) ]
            )
        
            logging.info('Pipeline for both numericak and categorical columns created')

            transformer = ColumnTransformer(
                [
                ('numerical_pipeline', numerical_pipeline, numerical_columns),
                ('categorical_pipeline', categorical_pipeline,categorical_columns )
                ]
            )

            return transformer
    except Exception as e:
        raise CustomException(e, sys)
    

    def feature_engineering(self, train_path, test_path):
        try:

            logging.info("Engineering in progress")

            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)

            logging.info("Done reading both train and test data")

            transformer = self.transfomer_object()
            logging.info('Done obtaining Transfomer object')
            
            target_variable = 'delivery_time'
            train_input_data = train_df.drop(columns=[target_variable], axis =1)
            test_input_data = test_df.drop(columns=[target_variable], axis =1)
            train_target_data = train_df[target_variable]
            test_target_data = test_df[target_variable]

            logging.info("Done splitting data into dependent and independent variables")


            input_train_array = transformer.fit_transform(train_input_data)
            input_test_array = transformer.transform(test_input_data)

            logging.info('Done Transforminng Input data')

            #Reshaping Target data
            output_train = np.array(train_target_data)
            output_test  = np.array(test_target_data)

            output_train_array= np.reshape(output_train, (-1, 1))
            output_test_array= np.reshape(output_test, (-1, 1))


            logging.info('Done converting train and test data into  array')

            return (
                input_train_array,
                input_test_array,
                output_train_array, 
                output_test_array
            )
        
        except Exception as e:
            raise CustomException(e, sys)
            
