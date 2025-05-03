# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from config import PROCESSED_DATA_PATH

def preprocess_data():
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Limpiar y mapear 'Target'
    df = df[df['Target'].notnull()]
    df['Target'] = df['Target'].map({'Dropout': 0, 'Graduate': 1})

    # Feature Engineering
    df['GPA_1st_sem'] = df['Curricular units 1st sem (grade)'] / 20
    df['Approval_rate_1st_sem'] = df['Curricular units 1st sem (approved)'] / df['Curricular units 1st sem (enrolled)']
    df['Approval_rate_1st_sem'] = df['Approval_rate_1st_sem'].replace([np.inf, -np.inf], np.nan).fillna(0)
    df['Economic_index'] = (df['Unemployment rate'] + df['Inflation rate']) / df['GDP']
    df['Economic_index'] = df['Economic_index'].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Separar características y objetivo
    X = df.drop(['Target', 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)'], axis=1)
    y = df['Target']

    # Identificar columnas categóricas y numéricas
    categorical_cols = [
        'Marital status', 'Application mode', 'Course', 'Nacionality',
        'Mother\'s qualification', 'Father\'s qualification',
        'Mother\'s occupation', 'Father\'s occupation'
    ]
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Pipeline de preprocesamiento
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

    X_processed = preprocessor.fit_transform(X)

    # ✅ Convertir a matriz densa antes de guardar
    X_processed = X_processed.toarray()  # ← Importante: convierte de sparse a dense

    # Aplicar SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_processed, y)

    # Guardar datos procesados
    pd.DataFrame(X_resampled).to_csv('data/processed/X_processed.csv', index=False)
    pd.Series(y_resampled).to_csv('data/processed/y.csv', index=False)
    print("✅ Datos procesados guardados en 'data/processed/'")