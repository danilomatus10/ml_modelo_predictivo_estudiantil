# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from config import PROCESSED_DATA_PATH
import os

def preprocess_data():
    try:
        print("🔍 Paso 1: Leyendo datos...")
        df = pd.read_csv(PROCESSED_DATA_PATH)

        # Validar que el archivo tenga datos
        if df.empty:
            raise ValueError("⚠️ El archivo 'cleaned_data.csv' está vacío.")

        print(f"📊 Dimensiones iniciales: {df.shape}")

        # Limpiar y mapear 'Target'
        print("🧹 Paso 2: Limpieza de Target...")
        # Convertir a string y limpiar espacios
        df['Target'] = df['Target'].astype(str).str.strip()
        
        # Mantener solo filas con valores válidos
        valid_labels = ['Dropout', 'Graduate']
        df = df[df['Target'].isin(valid_labels)]
        
        # Mapear a 0 y 1
        df['Target'] = df['Target'].map({'Dropout': 0, 'Graduate': 1})
        
        # Validar que y no tenga NaN
        if df['Target'].isnull().any():
            print("❌ Advertencia: Hay valores NaN en 'Target' después del mapeo")
            print("🔍 Valores únicos en 'Target':", df['Target'].unique())
            raise ValueError("La columna 'Target' contiene valores no válidos")

        print(f"✅ Valores únicos en Target después del mapeo: {df['Target'].unique()}")

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
        X_processed = X_processed.toarray()

        # Aplicar SMOTE
        print("🔁 Paso 7: Aplicando SMOTE...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_processed, y)

        # Crear directorio si no existe
        os.makedirs('data/processed', exist_ok=True)

        # Guardar datos procesados
        print("💾 Paso 8: Guardando datos procesados...")
        pd.DataFrame(X_resampled).to_csv('data/processed/X_processed.csv', index=False)
        pd.Series(y_resampled).to_csv('data/processed/y.csv', index=False)
        print("✅ Archivos generados correctamente en 'data/processed/'")

    except Exception as e:
        print(f"❌ Error durante el preprocesamiento: {str(e)}")
        raise

if __name__ == "__main__":
    preprocess_data()