import sys, os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from src.CustomerChurn.logger import logging
from src.CustomerChurn.exception import CustomException
from src.CustomerChurn.utils import save_object
from sklearn.compose import ColumnTransformer

@dataclass
class DataTransformationConfig:
    preprocessor_obj_filepath:str = os.path.join('artifacts','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
        
        
    def transformer_obj(self):
        
        try:
            target_cols = 'Churn'
            feature_cols = ['Age', 'Gender', 'Location', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB']
            col_to_encode = ['Gender', 'Location']
            
            logging.info('Pipeline creation started')
            
            logging.info(f"target col: {target_cols}")
            logging.info(f"feature cols: {feature_cols}")
            logging.info(f"cols that are to be encoded: {col_to_encode}")
            
            encoder_pipeline = Pipeline(steps=[
                ("encoding", OrdinalEncoder())
            ])
            
            preprocessor = ColumnTransformer([
                ("encoderpipe", encoder_pipeline, col_to_encode),
            ], remainder='passthrough')
            
            logging.info("Created transformer")
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)



    
    def initiate_data_transformation(self, train_path, test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Reading train and test data completed")
            
            logging.info("Obtaining preprocessing object")
            logging.info("initiating transformer pipeline")
            
            preprocessing_obj = self.transformer_obj()
            target_cols = 'Churn'
            feature_cols = ['Age', 'Gender', 'Location', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB']
            col_to_encode = ['Gender', 'Location']
            
            '''
            data = pd.DataFrame({'gender' : ['Female', 'Male','Female','Female','Male'], 
                            'state':['Chicago', 'Houston', 'Los Angeles', 'Miami', 'New York']})
            #print(data)
            #generating mappings
            
            encoder = OrdinalEncoder()
            encoder.fit_transform(data)
          
            for i, col in enumerate(encoder.categories_):
                print(f"\n{data.columns[i]}:")
                for j in col:
                    print(f"  {j[0]} -> {j[1]}")
                        
            print(data)'''
            
            
            # using preprocessor    
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            input_features_train_df = train_df.drop(columns=[target_cols], axis=1)
            target_feature_train_df = train_df[target_cols]
            
            input_feature_test_df = test_df.drop(columns=[target_cols], axis=1)
            target_feature_test_df = test_df[target_cols]
            
            print(input_features_train_df)
            #print(input_feature_test_df)
            
            logging.info("Applying Preprocessing on training and test dataframe")
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df )
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            print(input_feature_train_arr)
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            print(train_arr)
            
            logging.info(f"Saved preprocessing object")
            preprocessor_obj_filepath = self.data_transformation_config.preprocessor_obj_filepath
            save_object(

                file_path=preprocessor_obj_filepath ,
                obj=preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_filepath
            )
        except Exception as e:
            raise CustomException(e, sys)
