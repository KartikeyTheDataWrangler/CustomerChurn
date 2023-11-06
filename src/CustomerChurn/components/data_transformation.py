import sys, os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from src.CustomerChurn.logger import logging
from src.CustomerChurn.exception import CustomException
from src.CustomerChurn.utils import save_object
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

@dataclass
class DataTransformationConfig:
    preprocessor_obj_filepath:str = os.path.join('artifacts','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def transformer_obj(self):
        
        try:
            target_cols = 'churn'
            feature_cols = ['Age', 'Gender', 'Location', 'Subscription_Length_Months','Monthly_Bill',
                            'Total_Usage_GB']
            col_to_encode = ['Gender', 'Location']
            
            
            logging.info('Pipeline creation started')
            
            logging.info(f"target col : {target_cols}")
            logging.info(f"feature cols: {feature_cols}")
            logging.info(f"cols that are to be encoded: {col_to_encode}")
            
            
            #encoder = LabelEncoder()
            #mapping = dict(zip(encoder.classes_, 
                               #encoder.transform(encoder.classes_)))
            
            
            encoder_pipeline = Pipeline(steps=[
                ("encoding",LabelEncoder())
            ])
            
            preprocessor = ColumnTransformer([
                ("encoderpipe", encoder_pipeline, col_to_encode)
            ])
            
            logging.info("Created transformer")
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)

    
    def initiate_data_transformation(self, train_path, test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            print(train_df)
            print(test_df)
            logging.info("Reading train and test data completed")
            
            logging.info("Obtaining preprocessing object")
            logging.info("initiating transformer pipeline")
            
            preprocessing_obj = self.transformer_obj()
            target_cols = 'churn'
            feature_cols = ['Age', 'Gender', 'Location', 'Subscription_Length_Months','Monthly_Bill',
                            'Total_Usage_GB']
            col_to_encode = ['Gender', 'Location']
            
            label_encoder_mapping = {}
            
            
            for col in col_to_encode:
                label_encoder = LabelEncoder()
                train_df[col] = label_encoder.fit_transform(train_df[col])
                label_encoder_mapping[col] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))) 
            
            logging.info("Label encoding classes generated:")
            for col, encoding in label_encoder_mapping.items():
                logging.info(f"{col} : {encoding}")
            print(label_encoder_mapping)
        except:
            pass
            
            
        