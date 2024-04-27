import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.logger import logging
from src.utility_file import Utility
from datetime import datetime
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score


params = Utility().read_params()


class ModelTrainer:
    def __init__(self) -> None:
        pass

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise e

    def model_trainer(self, x_train, y_train, x_test, y_test):
        try:
            logging.info('Model Trainer')
            X_train=x_train
            y_train=y_train
            X_test=x_test
            y_test=y_test
            print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
            # print(y_train[:,1])
            # print('train 1')
            # print(y_train[:,0])
            # # print(X_test)
            
            models=[
                SVC(kernel='linear'),
                LinearSVC(),
                KNeighborsClassifier(),
                MultinomialNB(),

                ]
            perform_list = []
            
            for model in models:
                oneVsRest = OneVsRestClassifier(model)
                # print(oneVsRest)
                oneVsRest.fit(X_train, y_train)
                
                y_pred = oneVsRest.predict(X_test)

                # Performance metrics
                accuracy = accuracy_score(y_test, y_pred)
                # print(accuracy,oneVsRest)

                # Get precision, recall, f1 scores
                precision, recall, f1score, support = score(y_test, y_pred, average='micro')

                # Add performance parameters to list
                perform_list.append(dict([('Model', model),('Accuracy', accuracy),('Precision', precision),('Recall', recall),('F1', f1score)]))
            
            # Best model
            logging.info("Implementing hyper-parameters tuning")
            mn=MultinomialNB()#alpha=1.0, force_alpha=True, fit_prior=True)
            param = {'alpha': [0.01, 0.1, 0.5, 1.0, 10.0],  'fit_prior': [True, False],}
              
            clf = GridSearchCV(mn, param_grid=param, refit = True, verbose = 3, n_jobs=-1).fit(X_train, y_train)

            print('Best Parameters : {}'.format(clf.best_params_))
            #   print('best_params: ',clf.best_params_)
            print('Best Accuracy Grid Search : {}\n'.format(clf.best_score_))

            # Performance metrics
            Y_preds_train = clf.best_estimator_.predict(X_train)
            Y_preds = clf.best_estimator_.predict(X_test)
            
            # Get precision, recall, f1 scores
            clf_precision, clf_recall, clf_f1score, support = score(y_test, Y_preds, average='micro')

            # print("Test Accuracy Score : {:.3f}".format(accuracy_score(y_test, Y_preds)))
            # print("Train Accuracy Score : {:.3f}".format(accuracy_score(y_train, Y_preds_train)))
            print("\nClassification Report :")
            print(classification_report(y_test, Y_preds))
            class_report = classification_report(y_test, Y_preds, output_dict=True)
            class_report = pd.DataFrame(class_report).transpose()
            model_dir=params['DATA_LOCATION']['main_model_folder']
            report_path=params['DATA_LOCATION']['clf_report_filename']
            class_report.to_csv(os.path.join(model_dir,str(report_path)), index=False, sep=',')
            

            perform_list.append(dict([('Model', clf),('Accuracy', accuracy_score(y_train, Y_preds_train)),('Precision', clf_precision),('Recall', clf_recall),('F1', clf_f1score)]))
            
            newlist=pd.DataFrame(perform_list)
            newlist.to_csv('models\models_report_file.csv',index=False)

            joblib.dump(clf, 'models\grid_search_model.joblib')
            
        except Exception as e:
            logging.error(e)
            raise e
