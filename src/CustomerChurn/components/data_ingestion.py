import os
import sys
from src.CustomerChurn.exception import CustomException
from src.CustomerChurn.exception import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.CustomerChurn.components.data_transformation import DataTransformation






#creating paths for train, test and raw data

@dataclass
class DataIngestionConfig:
    train_data_path :str = os.path.join('artifacts','train.csv')        
    test_data_path : str = os.path.join('artifacts','test.csv')
    raw_data_path : str = os.path.join('artifacts','data.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("entered data ingestion method")
        
        try: 
            df = pd.read_excel(f'notebook\customer_churn_large_dataset.xlsx')
            logging.info('Read the raw data')
            df2 = df.drop(['CustomerID','Name'],axis=1)
            #print(df)
            logging.info('Drop cols the raw data')
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df2.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Train test split initiated")
            
            train_set,test_set=train_test_split(df2,test_size=0.3,random_state=12)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
            
        except Exception as e:
            raise CustomException(e,sys)


if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    datatransformation = DataTransformation()
    datatransformation.transformer_obj()
    datatransformation.initiate_data_transformation(train_data, test_data)
    
    
  