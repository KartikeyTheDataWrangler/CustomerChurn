
import pandas as pd
import numpy as np
import os
import sys
from src.CustomerChurn.logger import logging
from src.CustomerChurn.exception import customexception
from dataclasses import dataclass
from src.CustomerChurn.utils import save_object

from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier





@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class   ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting features and target from both training and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
        except:
            pass


class model_evaluate:
    def __init__(self):
        pass
    
              
    def save_multiple_models(self,X_train, y_train, X_test, y_test):
        
        try:
            
            ypred = {}
            self.models = [RandomForestClassifier(n_estimators=250), XGBClassifier(n_estimators=250,booster='gbtree',learning_rate=0.01), KNeighborsClassifier(n_neighbors=100),
                            MultinomialNB()]
            for m in self.models:
                model = m.fit(X_train, y_train)
                ypred[str(m)] = model.predict(X_test)
                
            prediction = pd.DataFrame.from_dict(ypred)
            print(prediction)
            summed_predictions = prediction.sum(axis=1)
            combined_predictions = (summed_predictions > 0).astype(int)
        except:
            pass    
            
        
