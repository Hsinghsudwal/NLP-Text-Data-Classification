from dataclasses import dataclass
import pandas as pd
import numpy as np
from src.logger import logging
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import joblib
import os
import sys
from src.utility_yaml_file import write_yaml_file, read_yaml_file

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_ingestion import MakeDataset
from src.data_validation import DataValidation
from src.utility_file import Utility

# from src.data_validation.DataValidation import validate_columns

params = Utility().read_params()


class DataPreprocess:
    def __init__(self) -> None:
        pass

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise e

    def data_validate(file_path):
        try:
            logging.info("Validating the data, before pre-process started")
            # report_path = read_yaml_file("artifacts/report.yaml")
            report_path = read_yaml_file(file_path)
            # dict_list=list(report_path.keys())
            dict_list_value = list(report_path.values())
            # print(dict_list_value[0])
            status = True
            if dict_list_value[0] == status:
                logging.info("Data validation is correct format")
                return True
            else:
                return False

        except Exception as e:
            logging.error(e)
            raise e

    def preprocess_data(self):

        try:
            # validate, clean the data
            # countVectorizer transform, save countVectorizer model to model folder

            path_dir = "models/report.yaml"
            DataPreprocess.data_validate(path_dir)
            logging.info("Validate data checked out")

            target = params["basic"]["TARGET_COLUMN"]
            category = params["basic"]["FEATURE_NAME"]
            artifact_dir = params["DATA_LOCATION"]["ARTIFACT_DIR"]
            raw_path = params["DATA_LOCATION"]["RAW_FILE_NAME"]
            train_path = params["DATA_LOCATION"]["TRAIN_FILE_NAME"]
            test_path = params["DATA_LOCATION"]["TRAIN_FILE_NAME"]
            model_dir = params["DATA_LOCATION"]["main_model_folder"]

            logging.info("Loading the data to perform preprocess")

            raw_data = DataPreprocess.read_data(
                os.path.join(artifact_dir, str(raw_path))
            )
            train_data = DataPreprocess.read_data(
                os.path.join(artifact_dir, train_path)
            )
            test_data = DataPreprocess.read_data(os.path.join(artifact_dir, test_path))

            # print(path_data)

            logging.info("training feature")

            # training frame
            input_feature_train_df = train_data[category]
            # print('train: ',input_feature_train_df.head())
            target_feature_train_df = train_data[target]
            input_feature_train_df=input_feature_train_df.apply(Utility.clean_text)

            # testing frame
            input_feature_test_df = test_data[category]
            target_feature_test_df = test_data[target]
            input_feature_train_df=input_feature_train_df.apply(Utility.clean_text)

            countvector = CountVectorizer(
                max_features=5000, analyzer="word", stop_words="english"
            )
            xtrain = countvector.fit_transform(input_feature_train_df)
            # print(x)
            ytrain = target_feature_train_df
            # print(y)

            # X_train,y_train,X_test,y_test=train_test_split(x, y,test_size=0.2, random_state=42)

            xtest = countvector.transform(input_feature_test_df)
            ytest = target_feature_test_df

            logging.info(f"{xtrain.shape, xtest.shape, ytrain.shape, ytest.shape}")

            logging.info("Saved preprocessing object.")

            # Utility().create_folder(model_dir)
            joblib.dump(countvector, "models\count_vector.joblib")

            return (xtrain, ytrain, xtest, ytest)

        except Exception as e:
            logging.error(e)
            raise e
