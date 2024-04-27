import pandas as pd
import os
import sys
from src.logger import logging
import pymongo
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from src.utility_file import Utility

from dotenv import load_dotenv
load_dotenv()

params = Utility().read_params()

class MakeDataset:

    def __init__(self) -> None:
        pass

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise e

    def dataset(self):

        try:
            logging.info('Retrieving data from mongodb')

            url= MongoClient(os.getenv('connection_to'))
            # print(url)

            #database
            db=url['NLP-Database']
            # print(db)

            #collection
            collec=db['NLP-Collection']
            x=collec.find()
            newlist=[]
            for data in x:
                newlist.append(data)
            # print(newlist)

            df=pd.DataFrame(newlist)
            # print(df.head())

            artifact_dir=params['DATA_LOCATION']['ARTIFACT_DIR']
            Utility().create_folder(artifact_dir)

            raw_file=params['DATA_LOCATION']['RAW_FILE_NAME']
            train_file=params['DATA_LOCATION']['TRAIN_FILE_NAME']
            test_file=params['DATA_LOCATION']['TEST_FILE_NAME']

            logging.info("Saving raw file to artifact folder -> raw.csv ")
            df.to_csv(os.path.join(artifact_dir, str(
                raw_file)), index=False, sep=',')
            
            logging.info("Saving train_set and test_set to artifact folder -> train.csv, test.csv")
            test_size=params['basic']['TEST_SIZE']
            train_set,test_set=train_test_split(df,test_size=test_size, random_state=42)

            train_data_path=os.path.join(artifact_dir,str(train_file))
            test_data_path=os.path.join(artifact_dir,str(test_file))

            # train_set.to_csv(os.path.join(artifact_dir,str(train_file)),index=False, sep=',')
            # test_set.to_csv(os.path.join(artifact_dir,str(test_file)), index=False, sep=',')
            train_set.to_csv(train_data_path, index=False, header=True)
            test_set.to_csv(test_data_path, index=False, header=True)

            logging.info("Data Ingestion has completed")

            return(train_data_path,test_data_path)
            
        except Exception as e:
            logging.error(e)
            raise e
