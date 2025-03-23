import pandas as pd
import numpy as np
import os 
import sys
from dataclasses import dataclass
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import mlflow
from urllib.parse import urlparse

@dataclass
class ModelTrainerConfig:
    model_path = os.path.join("artifacts", 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_config = ModelTrainerConfig()

    def evaluate_model(self, model, X_train, X_test, y_train,y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    def trin_best_model(self, X_train, X_test, y_train, y_test):
        models = {
            'RandomForest': RandomForestClassifier(),
            'LogisticRegression': LogisticRegression(),
            'SVM': SVC(kernel='rbf', random_state=42)
        }
    
        all_results = []

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train, y_train) # Train model
        
            # Evaluate Train and Test dataset
            accuracy = self.evaluate_model(model, X_train, X_test, y_train, y_test)
            all_results.append({'model_name': list(models.keys())[i], 'accuracy': accuracy, 'model': model})


        print()
        print(f'all_results - {all_results}')
        print()
        best_model = max(all_results, key=lambda x: x['accuracy'])
        print(f'model_list - {best_model}')
        

        mlflow.set_registry_uri('https://dagshub.com/iPatel007/py_practice1.mlflow')
        track_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # MLFlow code
        with mlflow.start_run():        
            model = best_model['model']    
            predict_resule = model.predict(X_test)
            accuracy = accuracy_score(y_test, predict_resule)

            mlflow.log_metric('accuracy', accuracy)
            if track_url_type_store != "file":
                mlflow.sklearn.log_model(model, "model", registered_model_name=best_model['model_name'])
            else:
                mlflow.sklearn.log_model(model, "model")


       

    def init_model_training(self, trian_df, test_df):
        try:
            X_train, y_train, X_test, y_test = (
                trian_df[:, :-1],
                trian_df[:, -1],
                test_df[:, :-1],
                test_df[:, -1]
            )

            self.trin_best_model(X_train, X_test, y_train, y_test)
        except Exception as e:
            raise CustomException(e, sys)