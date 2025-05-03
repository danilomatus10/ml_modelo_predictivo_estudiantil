# src/data_loader.py
import pandas as pd
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH

def load_and_clean_data():
    # Cargar datos
    df = pd.read_csv(RAW_DATA_PATH, delimiter=';', low_memory=False)
    
    # Corregir nombres de columnas
    df.columns = df.columns.str.replace('\n', '').str.strip()
    
    # Eliminar filas con valores faltantes críticos (>30%)
    threshold = len(df.columns) - int(0.3 * len(df.columns))
    df = df.dropna(thresh=threshold)
    
    # Guardar datos limpios
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Datos limpios guardados en {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    load_and_clean_data()