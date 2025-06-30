import os

PROJECT_PATH = os.environ.get('PROJECT_PATH', '.')

RAW_DATA_PATH = os.path.join(PROJECT_PATH, 'data/raw/MK_RAW_DATA.csv')
PROCESSED_DATA_PATH = os.path.join(PROJECT_PATH, 'data/processed/MK_PROCESSED_DATA.csv')