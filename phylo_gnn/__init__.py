import os
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv('DATA_PATH')
SEED = 50