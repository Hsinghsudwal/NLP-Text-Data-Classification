import os
import pandas as pd
import json
import pymongo
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()


def upload_mongodb(filename, database_name, collection_name):

    mongo_url= MongoClient(os.getenv("connection_to"))
    print(mongo_url)

    df = pd.read_csv(filename)
    json_data = list(json.loads(df.T.to_json()).values())
    print(' json data DONE')

    db=mongo_url[database_name]
    print("DATABASE: " , db)
    collection_db=db[collection_name]
    print('COLLECTIN NAME: ', collection_db)
    collection_db.insert_many(json_data)
    print('DATA INSERTED')

