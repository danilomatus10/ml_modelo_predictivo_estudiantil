# tests/test_preprocessing.py
# tests/test_preprocessing.py
import pandas as pd
from src.preprocessing import preprocess_data
import os

def test_preprocessing():
    print("🧪 Ejecutando pruebas de preprocesamiento...")
    
    # Eliminar archivos antiguos
    for f in ['X_processed.csv', 'y.csv']:
        p = os.path.join('data', 'processed', f)
        if os.path.exists(p):
            os.remove(p)
    
    # Ejecutar preprocesamiento
    preprocess_data()
    
    # Cargar datos procesados
    X = pd.read_csv('data/processed/X_processed.csv')
    y = pd.read_csv('data/processed/y.csv').squeeze()
    
    # Verificar dimensiones
    assert X.shape[0] > 0, "❌ X está vacío"
    assert y.shape[0] > 0, "❌ y está vacío"
    assert X.shape[0] == y.shape[0], "❌ Dimensiones inconsistentes entre X e y"
    
    # Verificar tipos numéricos
    assert X.dtypes.apply(lambda dtype: pd.api.types.is_numeric_dtype(dtype)).all(), "❌ X contiene valores no numéricos"
    print("✅ Todas las pruebas pasaron correctamente")

if __name__ == "__main__":
    test_preprocessing()