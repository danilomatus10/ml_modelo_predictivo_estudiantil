# src/data_loader.py
import pandas as pd
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH

def load_and_clean_data():
    # Cargar datos con delimitador y codificación correctos
    df = pd.read_csv(RAW_DATA_PATH, delimiter=';', low_memory=False, encoding='utf-8')
    
    # Corregir nombres de columnas (eliminar espacios y saltos de línea)
    df.columns = df.columns.str.replace('\n', '').str.strip()
    
    # Eliminar filas con valores faltantes críticos
    threshold = len(df.columns) - int(0.3 * len(df.columns))
    df = df.dropna(thresh=threshold)
    
    # ✅ Asegurarse de que 'Target' esté presente
    if 'Target' not in df.columns:
        raise ValueError("❌ Columna 'Target' no encontrada en los datos")
    
    # Guardar con delimitador correcto
    df.to_csv(PROCESSED_DATA_PATH, index=False, sep=';')  # ← Usa el mismo delimitador
    print(f"Datos limpios guardados en {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    load_and_clean_data()