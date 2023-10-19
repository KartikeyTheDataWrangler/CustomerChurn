import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.CustomerChurn.exception import CustomException
from src.CustomerChurn.logger import logging

from src.CustomerChurn.utils import save_object
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        #making preprocessor file
        try:
            col_to_drop = ["CustomerID","Name"]
            
            cat_col = ["Gender","Location"]
            
            num_col = ["Age", "Subscription_Length_Months", "Monthly_Bill", "Total_Usage_GB	"]
            
            #our dataset has no null values as determined in notebook 
            # we will only convert categorical columns to numerical columns
            
            cat_pipeline = Pipeline(
                steps=[
                    ("LabelEncoding",LabelEncoder())
                ]
            )
            
            logging.info(f"Categorical columns: {cat_col}")
            logging.info(f"Numerical columns: {num_col}")
            logging.info(f"Columns to drop: {col_to_drop}")
            
            preprocessor = ColumnTransformer(
                [("cat_pipeline",cat_pipeline, cat_col),
                 ("remainder", 'passthrough', num_col),
                 ("remainder", 'passthrough', col_to_drop)]
            )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data completed") 
            logging.info("Obtaining Preprocessing Object")
            
            preprocessing_obj=self.get_data_transformer_object()
            
            target_col_name = "Churn"
            other_col_drop = ["CustomerID","Name"]
            
           
            
            input_feature_train_df_ = train_df.drop(columns=[target_col_name],axis=1)
            
            input_feature_train_df = input_feature_train_df_.drop(columns=other_col_drop,axis=1)
            print(input_feature_train_df)
            target_feature_train_df=train_df[target_col_name]
            logging.info("train, feature and target created")
            
            
            
            input_feature_test_df_ = test_df.drop(columns=[target_col_name],axis=1)
            input_feature_test_df =  input_feature_test_df_.drop(columns=other_col_drop,axis=1)
            
            target_feature_test_df= test_df[target_col_name]
            logging.info("test, feature and target created")
            
            
            logging.info(f"Appliying Preprocessing obj on train and test df")
           
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            #print(input_feature_test_arr)
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
             
        except Exception as e:
            raise CustomException(e,sys)
        