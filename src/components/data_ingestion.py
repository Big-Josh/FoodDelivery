import sys
import os 
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    raw_data_path = os.path.join('artifacts','data.csv')
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self) :
        self.ingestion_config = DataIngestionConfig()

    def extract_data(self):
        logging.info('Extraction of Data is initiated')

        try:
            raw_data = pd.read_csv('DataWarehouse\historical_data.csv')
            logging.info('Read dataframe from DataWareHouse')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok= True)
            raw_data.to_csv(self.ingestion_config.raw_data_path, index =False, header = True)

            return self.ingestion_config.raw_data_path
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def clean_data(self,data_path):
        logging.info('Data Cleaning is Initiated')

        try:
            df = pd.read_csv(data_path)

            '''Creating delivery time column in seconds and creating other time columns'''
    
            df['created_at'] = pd.to_datetime(df['created_at'])
            df['actual_delivery_time'] = pd.to_datetime(df['actual_delivery_time'])
            df['delivery_time'] = (df.actual_delivery_time -df.created_at).astype('timedelta64[s]')
            df['delivery_time'] = df['delivery_time'] /  np.timedelta64(1, 's')
            df['hour_of_order'] = df['created_at'].dt.hour
            df['day_of_order'] = df['created_at'].dt.weekday

            logging.info('Done creating time colunmns')
            
            ''' Dropping outliers from delivery_time '''
            
            logging.info('Dropping of outliers')

            q1 = df['delivery_time'].quantile(0.25)
            q3 = df['delivery_time'].quantile(0.75)
            iqr = q3 - q1
            lower_limit = q1 - (1.5 * iqr)
            upper_limit = q3 + (1.5 * iqr)
            df = df[(df['delivery_time'] > lower_limit) & (df['delivery_time'] < upper_limit)]
            
            logging.info('Done dropping outliers')

            ''' Dropping Duplicate Observation'''
            df = df.drop_duplicates(subset = df.columns, keep = 'first')
            
            '''Dropping Null values'''
            df = df.dropna()
            
            '''Dropping Irrelevant Columns'''
            df = df.drop(columns = ['created_at', 'actual_delivery_time', 'store_id'], axis = 1)
            
            for col in df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist():
                df = df[df[col] > 0]
                
            '''Removing all prices less than a dollar'''
            for col in ['min_item_price', 'max_item_price', 'subtotal']:
                df = df[df[col] > 99]

            logging.info('Data Cleaning is done')   
            cleaned_df = df

            ''' Creating lables for top 30 categories in store_primary_category'''
            cusine = df['store_primary_category'].value_counts().rename_axis('unique_values').reset_index(name='counts')
            top_30_cat = list(cusine.head(30)['unique_values'].values)
            map_dict = {k: v for v, k in enumerate(top_30_cat)}
            df['cusine'] = df['store_primary_category'].map(map_dict)
            df['cusine'] = df['cusine'].fillna(30)
            df = df.drop(columns  = 'store_primary_category', axis = 1)

            logging.info('Done Creating labels for top 30 categories in store_primary_category')

            logging.info('Initiating Train Test Split')

            Train_df , Test_df = train_test_split(cleaned_df , test_size = 0.25, random_state = 27)

            Train_df.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            Test_df.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
            



if __name__ == "__main__":
    obj = DataIngestion()
    data_path = obj.extract_data()
    train_path, test_path = obj.clean_data(data_path)
    data_transformation = DataTransformation()
    train_arr, test_arr  =data_transformation.feature_engineering(train_path, test_path)
    # trainer = ModelTrainer()
    # report = trainer.initiate_model_trainer(train_arr,test_arr)
    print(train_arr.shape, test_arr.shape)