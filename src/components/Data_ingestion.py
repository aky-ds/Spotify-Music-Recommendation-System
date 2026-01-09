import os
from pathlib import Path
from src.exception import CustomException
from src.logger.logger import logging
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv(dotenv_path="C:/Users/AYAZ UL HAQ/Downloads/Compressed/MLOPS End to End Using CICD/.env")

@dataclass
class DataIngestionConfig:
    raw_data_path=os.path.join('artifacts','raw_data.csv')

class DataIngestion:
    logging.info('Data Ingestion have been started')
    def __init__(self):
        """
        Data Ingestion Constructor
        
        :param self: Description
        """
        self.data_ingest_config=DataIngestionConfig()
    
    def mongodb_data_loader(self,uri):
        """
        function  for mongodb data loader
        
        :param self: Description
        :param url: data url
        """
        client = MongoClient(uri)
        db = client["Spotifys_Data"]
        col = db["sptcollection"]
        return pd.DataFrame(list(col.find()))
   
    def Instantiate_dataingestion(self):
        """
        Data Ingestion Instantiated method
        """
        logging.info('Data Ingestion Have been started')
        try:
            df=self.mongodb_data_loader(os.getenv("MongoDBURL"))
            os.makedirs(os.path.dirname(self.data_ingest_config.raw_data_path), exist_ok=True)
            df.to_csv(self.data_ingest_config.raw_data_path,index=False)
            logging.info('Raw Data have been created')
            return df
        except Exception as e:
            raise CustomException(e)

if __name__=="__main__":
    datainegst=DataIngestion()
    datainegst.Instantiate_dataingestion()





