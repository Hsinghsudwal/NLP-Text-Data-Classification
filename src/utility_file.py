import yaml
import os
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet")
from nltk.stem import PorterStemmer

nltk.download("punkt")
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


class Utility:

    def __init__(self, params_path="params.yaml") -> None:
        self.params_path = params_path

    def create_folder(self, folder_name):

        try:
            # Creating a directory if it does not exist already
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

        except Exception as e:
            raise e

    def save_model(file_path, obj):
        """save model to the path folder with name"""
        try:
            dir_path = os.path.dirname(file_path)

            os.makedirs(dir_path, exist_ok=True)

            with open(file_path, "wb") as file_obj:
                pickle.dump(obj, file_obj)

        except Exception as e:
            raise e

    def load_model(file_path):

        try:
            with open(file_path, "rb") as file_obj:
                return pickle.load(file_obj)

        except Exception as e:
            raise e

    def read_params(self):

        try:
            # Reading params yaml file
            with open(self.params_path, "r") as params_file:
                params = yaml.safe_load(params_file)

        except Exception as e:
            raise e

        else:
            return params
        
    def read_yaml(file_path):
        try:
            with open(file_path, "r") as file:
                ryaml = yaml.safe_load(file)
            return ryaml
        
        except Exception as e:
            raise e
        
    def write_yaml_file(file_path, content):
     try:
            
        with open(file_path, "w") as file:
            yaml.dump(content, file)

     except Exception as e:
        raise e
     
    def read_yaml_file(file_path):
        try:

            with open(file_path, 'rb') as yaml_file:
             return yaml.safe_load(yaml_file)
        

        except Exception as e:
         raise e

    def clean_text(text):
        # clean = re.sub("[^a-zA-Z]", " ", str(text))
        clean = re.sub("'", "", str(text))
        clean = re.sub("(-\d|\W)+", " ", str(text))
        clean = clean.lower()
        clean = clean.split()
        clean = [word for word in clean if word not in stopwords.words("english")]
        clean = [word for word in clean if word not in list(string.punctuation)]
        clean = [lemmatizer.lemmatize(word) for word in clean]
        return " ".join(clean)

    # except Exception as e:
    #     raise e
