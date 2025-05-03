# src/train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from config import MODEL_PATH

def train_models():
    try:
        print("📥 Cargando datos procesados...")
        X = pd.read_csv('data/processed/X_processed.csv')
        y = pd.read_csv('data/processed/y.csv').squeeze()

        print(f"📊 Dimensiones de X: {X.shape}, y: {y.shape}")

        # Validar que todos los valores sean numéricos
        if not np.issubdtype(X.dtypes[0], np.number):
            raise ValueError("❌ X contiene valores no numéricos. Verifica el preprocesamiento.")

        print("🧮 Dividiendo datos en entrenamiento y prueba...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # Entrenar Random Forest con GridSearchCV
        print("🌳 Entrenando Random Forest...")
        param_grid_rf = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'min_samples_split': [2, 5]
        }

        grid_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5, scoring='f1', n_jobs=-1)
        grid_rf.fit(X_train, y_train)

        print(f"✅ Mejor Random Forest: {grid_rf.best_params_}")
        y_pred_rf = grid_rf.predict(X_test)
        print("📊 Reporte de clasificación (Random Forest):")
        print(classification_report(y_test, y_pred_rf))

        # Entrenar XGBoost con GridSearchCV
        print("🚀 Entrenando XGBoost...")
        param_grid_xgb = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.01, 0.1]
        }

        grid_xgb = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), param_grid_xgb, cv=5, scoring='f1', n_jobs=-1)
        grid_xgb.fit(X_train, y_train)

        print(f"✅ Mejor XGBoost: {grid_xgb.best_params_}")
        y_pred_xgb = grid_xgb.predict(X_test)
        print("📊 Reporte de clasificación (XGBoost):")
        print(classification_report(y_test, y_pred_xgb))

        # Guardar mejor modelo
        best_model = grid_xgb
        joblib.dump(best_model, MODEL_PATH)
        print(f"💾 Modelo guardado en {MODEL_PATH}")

    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {str(e)}")
        raise

if __name__ == "__main__":
    train_models()