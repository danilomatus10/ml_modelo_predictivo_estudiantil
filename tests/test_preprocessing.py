# tests/test_preprocessing.py
import pandas as pd
import numpy as np
from src.preprocessing import preprocess_data


def test_preprocessing():
    print("🧪 Ejecutando pruebas de preprocesamiento...")
    
    # Eliminar archivos antiguos si existen
    import os
    if os.path.exists('data/processed/X_processed.csv'):
        os.remove('data/processed/X_processed.csv')
    if os.path.exists('data/processed/y.csv'):
        os.remove('data/processed/y.csv')
    
    # Ejecutar preprocesamiento
    preprocess_data()
    
    # Cargar datos procesados
    X = pd.read_csv('data/processed/X_processed.csv')
    y = pd.read_csv('data/processed/y.csv').squeeze()
    
    # Verificar dimensiones
    assert X.shape[0] > 0, "X está vacío"
    assert y.shape[0] > 0, "y está vacío"
    
    # Verificar tipos numéricos
    assert X.dtypes.apply(lambda dtype: pd.api.types.is_numeric_dtype(dtype)).all(), "X contiene valores no numéricos"
    
    # Verificar distribución de clases
    class_distribution = pd.Series(y).value_counts()
    assert class_distribution.min() > 0, "Una clase está vacía después de SMOTE"
    
    print("✅ Todas las pruebas de preprocesamiento pasaron")

if __name__ == "__main__":
    test_preprocessing()