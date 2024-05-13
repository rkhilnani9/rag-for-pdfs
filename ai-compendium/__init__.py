import os
from config import MODEL_API_KEY, MONGO_URI, MONGO_DB_NAME, MONGO_COLLECTION_NAME
from pymongo import MongoClient

os.environ["OPENAI_API_KEY"] = MODEL_API_KEY

# connect mongo db
client = MongoClient(MONGO_URI)
mongodb = client[MONGO_DB_NAME]
mongo_collection = mongodb[MONGO_COLLECTION_NAME]
