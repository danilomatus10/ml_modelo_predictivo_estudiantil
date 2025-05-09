# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from config import PROCESSED_DATA_PATH
import os
import sys

# Añadir el directorio raíz al path de Python
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def preprocess_data():
    """Procesa los datos limpios y genera conjuntos de entrenamiento con validación completa"""
    try:
        print("🔍 Paso 1: Leyendo datos...")
        df = pd.read_csv(PROCESSED_DATA_PATH, delimiter=';', low_memory=False)
        
        # Mostrar diagnóstico
        print(f"📊 Columnas en cleaned_data.csv: {df.columns.tolist()}")
        
        # Validar que 'Target' exista
        if 'Target' not in df.columns:
            raise KeyError("❌ Columna 'Target' no encontrada en los datos procesados")
        
        # Corregir nombres de columnas
        df.columns = df.columns.str.replace('\n', '').str.strip()
        
        # Validar que el archivo tenga datos
        if df.empty:
            raise ValueError("⚠️ El archivo 'cleaned_data.csv' está vacío.")
            
        print(f"📊 Dimensiones iniciales: {df.shape}")
        print(f"🔢 Cantidad de columnas numéricas: {len(df.select_dtypes(include=[np.number]).columns)}")
        print(f"🔤 Cantidad de columnas categóricas: {len(df.select_dtypes(include=['object']).columns)}")

        # Limpiar y mapear 'Target'
        print("🧹 Paso 2: Limpieza de Target...")
        df['Target'] = df['Target'].astype(str).str.strip()
        valid_labels = ['Dropout', 'Graduate']
        df = df[df['Target'].isin(valid_labels)]
        df['Target'] = df['Target'].map({'Dropout': 0, 'Graduate': 1})

        if df['Target'].isnull().any():
            print("❌ Advertencia: Hay valores NaN en 'Target' después del mapeo")
            print("🔍 Valores únicos en 'Target':", df['Target'].unique())
            raise ValueError("La columna 'Target' contiene valores no válidos")

        print(f"✅ Valores únicos en Target después del mapeo: {df['Target'].unique()}")

        # Feature Engineering
        print("🛠️ Paso 3: Feature Engineering...")
        # GPA normalizado
        df['GPA_1st_sem'] = df['Curricular units 1st sem (grade)'] / 20
        
        # Tasa de aprobación
        df['Approval_rate_1st_sem'] = df['Curricular units 1st sem (approved)'] / df['Curricular units 1st sem (enrolled)']
        df['Approval_rate_1st_sem'] = df['Approval_rate_1st_sem'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Índice económico
        df['Economic_index'] = (df['Unemployment rate'] + df['Inflation rate']) / df['GDP']
        df['Economic_index'] = df['Economic_index'].replace([np.inf, -np.inf], np.nan).fillna(0)

        # Separar características y objetivo
        print("🧮 Paso 4: Separando X y y...")
        X = df.drop(['Target', 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)'], axis=1)
        y = df['Target']

        # Identificar columnas categóricas y numéricas
        print("🔍 Paso 5: Identificando columnas categóricas y numéricas...")
        categorical_cols = [
            'Marital status', 'Application mode', 'Course', 'Nacionality',
            "Mother's qualification", "Father's qualification",
            "Mother's occupation", "Father's occupation"
        ]
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        print(f"🔢 Columnas numéricas: {numeric_cols}")
        print(f"🔤 Columnas categóricas: {categorical_cols}")

        # Pipeline de preprocesamiento
        print("🛠️ Paso 6: Aplicando preprocesamiento...")
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

        X_processed = preprocessor.fit_transform(X)

        # ✅ Convertir a matriz densa antes de guardar
        print(".Dense paso 7: Convirtiendo a matriz densa...")
        X_processed = X_processed.toarray()  # ← Importante: convierte de sparse a dense

        # Aplicar SMOTE
        print("🔁 Paso 8: Aplicando SMOTE para balancear clases...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_processed, y)

        # Validar que todos los valores sean numéricos
        print("🧮 Paso 9: Validando tipos numéricos...")
        if not np.issubdtype(X_resampled.dtype, np.number):
            raise ValueError("❌ X_resampled contiene valores no numéricos después de .toarray()")

        # Crear directorio si no existe
        os.makedirs('data/processed', exist_ok=True)

        # Guardar datos procesados
        print("💾 Paso 10: Guardando datos procesados...")
        pd.DataFrame(X_resampled).to_csv('data/processed/X_processed.csv', index=False)
        pd.Series(y_resampled).to_csv('data/processed/y.csv', index=False)
        print("✅ Archivos generados correctamente en 'data/processed/'")

    except Exception as e:
        print(f"❌ Error durante el preprocesamiento: {str(e)}")
        raise

if __name__ == "__main__":
    preprocess_data()