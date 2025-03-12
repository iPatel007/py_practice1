import os
from dataclasses import dataclass

from src.mlproject.utils import read_sql_data
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
import sys

from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join("artifacts", 'train.csv')
    test_data_path = os.path.join("artifacts", 'test.csv')
    raw_data_path = os.path.join("artifacts", 'raw.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()    
        
    def initiate_data_ingestion(self):
        try:
            logging.info("Reading from mysql database")
            df = read_sql_data()
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            #Save all data as raw.csv file in the artifacts/ dir
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            #train_test_split
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=11)
            
            #Save train and test DataFrame
            train_df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingesion is complited")

            return {
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            }

        except Exception as e:
            raise CustomException(e, sys)
