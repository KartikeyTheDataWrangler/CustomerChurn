import os
import sys

import numpy as np 
import pandas as pd
import pickle
from src.CustomerChurn.exception import CustomException
from sklearn.base import BaseEstimator, TransformerMixin


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

class Col_Dropper(BaseEstimator, TransformerMixin):
    def __init__(self, coltodrop):
        self.coltodrop = coltodrop
        
    def fit(self,X,y=None):
        Z = X.copy()
        return Z 
    
    
    def transform(self, X,y=None):
        
        Z = Z.drop(columns=self.coltodrop, axis=1)
        return Z
    
    
    
