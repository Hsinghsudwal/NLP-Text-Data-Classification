import os
import sys

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mongodb.mongodb_file import upload_mongodb
from src.data_ingestion import MakeDataset
from src.data_validation import DataValidation
from src.data_preprocess import DataPreprocess
from src.model_trainer import ModelTrainer
from src.model_mlflow import Modelflow
from sklearn.naive_bayes import MultinomialNB
from src.predict_pipeline import Predict
from prefect import flow


# @flow
def main_flow():
    # file = r"notebooks\bbc-text.csv"
    # database = "python_nlp_database"
    # collection = "python_nlp_collection"
    # upload_mongodb(file, database, collection)

    md = MakeDataset()
    md.dataset()

    dv = DataValidation()
    dv.validate()

    dp = DataPreprocess()
    xtrain, ytrain, xtest, ytest = dp.preprocess_data()

    mt = ModelTrainer()
    mt.model_trainer(xtrain, ytrain, xtest, ytest)

    mf = Modelflow()
    mf.model_flow(xtrain, ytrain, xtest, ytest)

    # Testing prediction return
    # pp=Predict()
    # pp.predict_pipeline()

if __name__ == "__main__":
    main_flow()

    
