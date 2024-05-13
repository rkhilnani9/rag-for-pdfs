import os

BASE_DIR = os.path.dirname(__file__)

DEBUG = False
PDF_TEXT_FILE_PATH = os.path.join(BASE_DIR, "pdf_text_2.txt")
MODEL_API_KEY = "******"
ANSWERING_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDINGS_SAVE_PATH = os.path.join(
    BASE_DIR, "temp_embeddings.csv"
)
MAX_TOKENS = 1600
BATCH_SIZE = 50
TOP_K = 2

# mongodb
MONGO_URI = "mongodb://localhost:27017"
MONGO_DB_NAME = "benefit_search"
MONGO_COLLECTION_NAME = "embeddings"
