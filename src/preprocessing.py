import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from src.config import COLUMN_MAP

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza robusta de nombres de columnas"""
    # Paso 1: Limpieza básica
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r'[\W_]+', '_', regex=True)  # Manejar caracteres especiales
        .str.replace('__+', '_', regex=True)
        .str.strip('_')
    )
    
    # Paso 2: Renombrar columnas específicas
    return df.rename(columns=COLUMN_MAP)

class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.numeric_features = [
            'previous_qualification_grade',
            'admission_grade',
            'age_at_enrollment'
        ]
        
        self.categorical_features = [
            'marital_status',
            'international',
            'gender'
        ]
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), self.numeric_features),
                
                ('cat', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore'))
                ]), self.categorical_features)
            ])

    def fit(self, X, y=None):
        X_clean = clean_columns(X)
        self.preprocessor.fit(X_clean)
        return self
    
    def transform(self, X):
        X_clean = clean_columns(X)
        return self.preprocessor.transform(X_clean)