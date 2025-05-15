# src/data_loader.py
import pandas as pd
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH

def load_and_clean_data():
    print("📥 Cargando y limpiando datos...")
    
    # Cargar datos con delimitador correcto
    df = pd.read_csv(RAW_DATA_PATH, delimiter=';', low_memory=False)
    
    # Corregir nombres de columnas
    df.columns = df.columns.str.replace('\n', '').str.strip()
    
    # Eliminar filas con >30% de valores faltantes
    threshold = len(df.columns) - int(0.3 * len(df.columns))
    df = df.dropna(thresh=threshold)
    
    # Imputar valores faltantes en columnas numéricas con mediana
    numeric_cols = df.select_dtypes(include=[float, int]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Validar que 'Target' esté presente
    if 'Target' not in df.columns:
        raise ValueError("❌ Columna 'Target' no encontrada")
    
    # Guardar datos limpios con delimitador ';'
    df.to_csv(PROCESSED_DATA_PATH, index=False, sep=';', encoding='utf-8')
    print(f"Datos limpios guardados en {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    load_and_clean_data()