# src/train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV  # ← Importación añadida
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from src.config import MODEL_PATH

def train_models():
    try:
        print("📥 Cargando datos procesados...")
        X = pd.read_csv('data/processed/X_processed.csv')
        y = pd.read_csv('data/processed/y.csv').squeeze()
        
        # Validar que X sea completamente numérico
        if not X.dtypes.apply(lambda dtype: pd.api.types.is_numeric_dtype(dtype)).all():
            raise ValueError("❌ X contiene valores no numéricos. Revisa el preprocesamiento.")
        
        print(f"📊 Dimensiones de X: {X.shape}, y: {y.shape}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # Entrenar Random Forest con GridSearchCV
        print("🧮 Entrenando Random Forest...")
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

        # Entrenar XGBoost con Bayesian Search (si lo tienes implementado)
        try:
            from skopt import BayesSearchCV
            from skopt.space import Real, Integer

            print("🚀 Entrenando XGBoost con Bayesian Optimization...")
            param_grid_bayes = {
                'n_estimators': Integer(50, 200),
                'max_depth': Integer(3, 10),
                'learning_rate': Real(0.01, 0.3, prior='log-uniform')
            }

            bayes_search = BayesSearchCV(
                XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                param_grid_bayes,
                n_iter=30,
                cv=5,
                scoring='f1',
                n_jobs=-1
            )
            bayes_search.fit(X_train, y_train)

            print(f"✅ Mejor XGBoost (Bayesian): {bayes_search.best_params_}")
            y_pred_xgb = bayes_search.predict(X_test)
            print("📊 Reporte de clasificación (XGBoost):")
            print(classification_report(y_test, y_pred_xgb))

            # Guardar mejor modelo
            joblib.dump(bayes_search, MODEL_PATH)
            print(f"💾 Modelo guardado en {MODEL_PATH}")

        except ImportError:
            print("⚠️ BayesSearchCV no está disponible. Asegúrate de tener scikit-optimize instalado.")

    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {str(e)}")
        raise

if __name__ == "__main__":
    train_models()