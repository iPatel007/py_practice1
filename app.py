from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
import sys
from src.mlproject.components.data_ingestion import DataIngestion

from src.mlproject.components.data_transformation import DataTransformation


if __name__ == "__main__":
    try:
        data_ingesion = DataIngestion()
        train_data_path, test_data_path = data_ingesion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        data_transformation.init_data_transformation(train_data_path, test_data_path)
        
    except Exception as e:
        raise CustomException(e, sys)
    

    
