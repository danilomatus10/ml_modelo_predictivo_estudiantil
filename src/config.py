# config.py
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(ROOT_DIR, 'data', 'raw', 'data.csv')
PROCESSED_DATA_PATH = os.path.join(ROOT_DIR, 'data', 'processed', 'cleaned_data.csv')
MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'best_model.pkl')