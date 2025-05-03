# tests/test_preprocessing.py
import pandas as pd
from src.preprocessing import preprocess_data

def test_preprocessing():
    preprocess_data()
    X = pd.read_csv('data/processed/X_processed.csv')
    y = pd.read_csv('data/processed/y.csv')
    
    assert not X.isnull().any().any(), "Hay valores faltantes en X"
    assert not y.isnull().any().any(), "Hay valores faltantes en y"
    assert X.shape[0] > 0, "X está vacío"
    assert y.shape[0] > 0, "y está vacío"
    print("Todas las pruebas pasaron")

if __name__ == "__main__":
    test_preprocessing()