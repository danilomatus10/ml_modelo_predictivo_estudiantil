# tests/test_preprocessing.py
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from src.preprocessing import preprocess_data

# Añadir la raíz del proyecto a la ruta de Python
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_preprocessing():
    """Prueba completa del preprocesamiento"""
    print("🧪 Iniciando pruebas de preprocesamiento...")
    
    # Eliminar archivos antiguos si existen
    processed_dir = Path('data/processed')
    for file in ['X_processed.csv', 'y.csv']:
        file_path = processed_dir / file
        if file_path.exists():
            file_path.unlink()
            print(f"🗑️ Eliminado archivo antiguo: {file}")

    # Ejecutar preprocesamiento
    print("🔄 Ejecutando preprocess_data()...")
    try:
        preprocess_data()
    except Exception as e:
        pytest.fail(f"❌ Error durante preprocesamiento: {str(e)}")

    # Cargar datos procesados
    print("📥 Cargando datos procesados...")
    X = pd.read_csv('data/processed/X_processed.csv')
    y = pd.read_csv('data/processed/y.csv').squeeze()

    # Verificar que los archivos existan
    assert Path('data/processed/X_processed.csv').exists(), "❌ X_processed.csv no se creó"
    assert Path('data/processed/y.csv').exists(), "❌ y.csv no se creó"

    # Verificar dimensiones
    assert X.shape[0] > 0, "❌ X está vacío"
    assert y.shape[0] > 0, "❌ y está vacío"
    assert X.shape[0] == y.shape[0], "❌ Dimensiones inconsistentes entre X e y"

    # Verificar tipos numéricos
    def is_numeric(col):
        try:
            pd.to_numeric(col, errors='coerce').notnull().all()
            return True
        except:
            return False
            
    non_numeric_cols = [col for col in X.columns if not is_numeric(X[col])]
    assert len(non_numeric_cols) == 0, f"❌ Columnas no numéricas: {non_numeric_cols}"

    # Verificar distribución de clases
    class_distribution = pd.Series(y).value_counts()
    assert class_distribution.min() > 0, "❌ Una clase está vacía después de SMOTE"
    
    print("✅ Todas las pruebas pasaron correctamente")

if __name__ == "__main__":
    test_preprocessing()