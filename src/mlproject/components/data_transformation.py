import os
import sys
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from dataclasses import dataclass
import pandas as pd
from src.mlproject.utils import read_sql_data   
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np

from src.mlproject.utils import save_preprocessor_object


@dataclass
class DataTransformationConfig:
    preprocessor_object_file_path = os.path.join("artifacts", 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.tranformation_config = DataTransformationConfig() 

    def get_data_transformation(self):
        try:
            logging.info("Data Transformation started")

            #Read the raw data and return the preprocessor object
            main_df = pd.read_csv('/Users/ipatel/Documents/Amit/Python/Practice/p1/artifacts/train.csv')
            main_df.drop(columns=['Survived'], axis=1, inplace=True)
            
            numeric_columns = main_df.select_dtypes(exclude=['object']).columns
            categorical_columns = main_df.select_dtypes(include=['object']).columns

            print(f"numeric_columns", numeric_columns)
            print(f"categorical_columns", categorical_columns)

            num_imputer = SimpleImputer(strategy='mean')
            cat_imputer = SimpleImputer(strategy='most_frequent')   
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            scaler = StandardScaler()

            num_pip = Pipeline(steps=[
                ('num_imputer', num_imputer),
                ('num_scaler', scaler)
            ])

            cat_pip = Pipeline(steps=[
                ('cat_imputer', cat_imputer),
                ('ohe', ohe),
                ("cat_scaler",StandardScaler(with_mean=False))

            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_pip, numeric_columns),
                    ('cat', cat_pip, categorical_columns)
                ]
            )

            return preprocessor
                    
        except Exception as e:
            raise CustomException(e, sys)       
        
    def init_data_transformation(self, train_path, test_path):
        try:
            #Read the train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            #Get the preprocessor object and save pkl file
            preprocessor = self.get_data_transformation()

            target_column = 'Survived'
            X_train = train_df.drop(columns=[target_column], axis=1)
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column], axis=1)
            y_test = test_df[target_column]

            input_train_arr = preprocessor.fit_transform(X_train)
            input_test_arr = preprocessor.transform(X_test)

            final_train_df = np.c_[input_train_arr, np.array(y_train)]
            final_test_df = np.c_[input_test_arr, np.array(y_test)]

           
            #Save the preprocessor object
            save_preprocessor_object(file_path=self.tranformation_config.preprocessor_object_file_path, 
                                     object=preprocessor)

            logging.info("Data Transformation completed")
            return (
                final_train_df,
                final_test_df,
                self.tranformation_config.preprocessor_object_file_path)
        
        except Exception as e:
            raise CustomException(e, sys)        