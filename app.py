from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
import sys
from src.mlproject.components.data_ingestion import DataIngestion


if __name__ == "__main__":
    try:
        data_ingesion = DataIngestion()
        data_ingesion.initiate_data_ingestion()
        
    except Exception as e:
        raise CustomException(e, sys)
    

    
