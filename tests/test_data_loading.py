# tests/test_data_loading.py
import os
import sys
from pathlib import Path

# Añadir el directorio raíz al path de Python
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Ahora puedes importar desde src/
from src.data_loader import load_and_clean_data
import pandas as pd

def test_data_loading():
    print("🧪 Ejecutando pruebas de carga de datos...")
    
    # Eliminar archivos antiguos si existen
    if os.path.exists('data/processed/cleaned_data.csv'):
        os.remove('data/processed/cleaned_data.csv')
    
    # Ejecutar carga de datos
    load_and_clean_data()
    
    # Cargar datos limpios
    df = pd.read_csv('data/processed/cleaned_data.csv', delimiter=';', low_memory=False)
    
    # Verificar dimensiones
    assert df.shape[0] > 0, "❌ cleaned_data.csv está vacío"
    
    # Verificar columnas críticas
    critical_columns = ['Target', 'Age at enrollment', 'Admission grade']
    for col in critical_columns:
        assert col in df.columns, f"❌ {col} no está en cleaned_data.csv"
    
    print("✅ Todas las pruebas pasaron correctamente")

if __name__ == "__main__":
    test_data_loading()