import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from src.config import RAW_DATA, MODELS_DIR
from src.preprocessing import DataPreprocessor, clean_columns

def load_and_validate_data():
    """Carga y valida estructura básica de los datos"""
    df = pd.read_csv(RAW_DATA, encoding='utf-8', skipinitialspace=True)
    df = clean_columns(df)
    
    # Validación crítica
    required_columns = {'target', 'previous_qualification_grade', 'admission_grade'}
    missing = required_columns - set(df.columns)
    if missing:
        raise KeyError(f'Columnas esenciales faltantes: {missing}')
    
    return df

def main():
    # 1. Cargar y validar datos
    df = load_and_validate_data()
    df['target'] = df['target'].str.strip().replace({'': None}).dropna()
    
    # 2. Preparar datos
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 3. Pipeline de modelado
    pipeline = Pipeline([
        ('preprocessing', DataPreprocessor()),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            random_state=42
        ))
    ])
    
    # 4. Entrenar y evaluar
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # 5. Guardar modelo
    joblib.dump(pipeline, MODELS_DIR / 'best_model.pkl')
    print(f'Modelo guardado en: {MODELS_DIR/"best_model.pkl"}')

if __name__ == "__main__":
    main()