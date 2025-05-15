# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from src.config import PROCESSED_DATA_PATH
import os

def preprocess_data():
    try:
        print("🔍 Paso 1: Leyendo datos...")
        # Cargar con delimitador correcto
        df = pd.read_csv(PROCESSED_DATA_PATH, delimiter=';', low_memory=False)
        
        # Diagnóstico de columnas
        print("📊 Columnas en cleaned_data.csv:", df.columns.tolist())
        print("🔍 Primeras filas:")
        print(df.head().to_string())

        # Validar que 'Target' esté presente
        if 'Target' not in df.columns:
            raise KeyError("❌ Columna 'Target' no encontrada")

        # Limpiar y mapear 'Target'
        print("🧹 Paso 2: Limpieza de Target...")
        df['Target'] = df['Target'].astype(str).str.strip()
        valid_labels = ['Dropout', 'Graduate']
        df = df[df['Target'].isin(valid_labels)]
        df['Target'] = df['Target'].map({'Dropout': 0, 'Graduate': 1})

        if df['Target'].isnull().any():
            print("❌ Advertencia: Valores NaN en 'Target'")
            raise ValueError("La columna 'Target' contiene valores no válidos")

        # Feature Engineering
        print("🛠️ Paso 3: Feature Engineering...")
        df['GPA_1st_sem'] = df['Curricular units 1st sem (grade)'] / 20
        df['Approval_rate_1st_sem'] = df['Curricular units 1st sem (approved)'] / df['Curricular units 1st sem (enrolled)']
        df['Approval_rate_1st_sem'] = df['Approval_rate_1st_sem'].replace([np.inf, -np.inf], np.nan).fillna(0)
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

        # Pipeline de preprocesamiento
        print("🛠️ Paso 6: Aplicando preprocesamiento...")
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

        X_processed = preprocessor.fit_transform(X)

        # ✅ Conversión a matriz densa
        X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X_processed.toarray(), y)

        # Validar que todos los valores sean numéricos
        if not np.issubdtype(X_resampled.dtype, np.number):
            raise ValueError("❌ X_resampled contiene valores no numéricos")

        # Crear directorio si no existe
        os.makedirs('data/processed', exist_ok=True)

        # Guardar datos procesados
        print("💾 Paso 7: Guardando datos procesados...")
        pd.DataFrame(X_resampled).to_csv('data/processed/X_processed.csv', index=False)
        pd.Series(y_resampled).to_csv('data/processed/y.csv', index=False)
        print("✅ Archivos generados correctamente en 'data/processed/'")

    except Exception as e:
        print(f"❌ Error durante el preprocesamiento: {str(e)}")
        raise

if __name__ == "__main__":
    preprocess_data()