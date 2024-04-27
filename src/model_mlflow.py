import pandas as pd
import json
import joblib
import os
import sys
from datetime import datetime
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score,precision_score,recall_score,log_loss
from urllib.parse import urlparse
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
from sklearn.metrics import accuracy_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.logger import logging
from src.utility_file import Utility

params = Utility().read_params()


class Modelflow:

    def __init__(self) -> None:
        pass

    # def model_experimentation(classifier, model_name, run_name):
    def model_flow(self, x_train, y_train, x_test, y_test):
        try:
            X_train = x_train
            y_train = y_train
            X_test = x_test
            y_test = y_test
            
            # X_train, y_train, X_test, y_test=data_pass(X_train, y_train, X_test, y_test)

            classifier = MultinomialNB()
            run_name = "NaiveBayes_model"

            experiment_name = "_NLP_text_classification_mlflow_experiments_"+ str(datetime.now().strftime("%m-%d-%y"))

            mlflow.set_experiment(experiment_name)
            experiment = mlflow.get_experiment_by_name(experiment_name)

            print("Name: {}".format(experiment_name))
            with mlflow.start_run(run_name=run_name, experiment_id=experiment.experiment_id):

                mlflow.sklearn.autolog()  # Logged information precission score, f1 score, recoll)

                param = {
                    "alpha": [0.01, 0.1, 0.5, 1.0, 10.0],
                    "fit_prior": [True, False],
                }

                clf = GridSearchCV(
                    classifier, param_grid=param, refit=True, verbose=3, n_jobs=-1
                )
                clf.fit(X_train, y_train)
                mlflow.log_params(clf.best_params_)
                
                y_pred = clf.predict(X_test)
                y_pred_test= clf.predict_proba(X_test)

                # mlflow.log_metrics(clf.best_score_)

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average="micro")
                recall = recall_score(y_test, y_pred, average="micro")
                f1score = f1_score(y_test, y_pred, average="micro")
                entropy = log_loss(y_test, y_pred_test)

                mlflow.log_metric("accuracy_score", accuracy)
                mlflow.log_metric("precision_score", precision)
                mlflow.log_metric("recall_score", recall)
                mlflow.log_metric("f1_score", f1score)
                mlflow.log_metric("log_loss", entropy)

                # Save model
                joblib.dump(clf, 'models\mlflow_model.joblib')

                ## For Remote server only(DAGShub)

                # remote_server_uri="https://dagshub.com/krishnaik06/mlflowexperiments.mlflow"
                # mlflow.set_tracking_uri(remote_server_uri)

                # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme


                # Model registry does not work with file store
                # if tracking_url_type_store != "file":
                # mlflow.sklearn.log_model(
                #         sk_model=svcc,
                #         artifact_path="sklearn-model",
                #         signature=signature,
                #         registered_model_name="NB-uri",)
        
            
                # else:
                mlflow.sklearn.log_model(clf, "model")
                # mlflow.sklearn.log_model(
                #         sk_model=svcc,
                #         artifact_path="sklearn-model",
                #         signature=signature,
                #         registered_model_name="SVC-model",)

        except Exception as e:
            logging.error(e)
            raise e

