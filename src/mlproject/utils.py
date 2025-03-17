import os
import sys
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
import pandas as pd
from dotenv import load_dotenv

import pickle

load_dotenv()

host = os.getenv('host')
user = os.getenv('host')
password = os.getenv('host')
db = os.getenv('host')

def read_sql_data():
    logging.info('Reading MySql database started')
    try:
        # myDB = pymysql.connect(
        #     host=host,
        #     user=user,
        #     passwd=password,
        #     db=db
        # )
        # logging.info('DB Connection Eshtablished', myDB)
        # df = pd.read_sql_query('Select * from student', myDB)
        # print(df.head())
        df = pd.read_csv('/Users/ipatel/Documents/Amit/Python/Practice/p1/cvs_files/train.csv')
        return df
    except Exception as e:
        raise CustomException(e, sys)
    

def save_preprocessor_object(file_path, object):
    logging.info("Saving preprocessor object")
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(object, file_obj)

    except Exception as e:
        raise CustomException(e, sys)    