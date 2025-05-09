# src/train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import joblib
from config import MODEL_PATH

def train_models():
    X = pd.read_csv('data/processed/X_processed.csv')
    
    # Validar que todas las columnas sean numéricas
    if not X.dtypes.apply(lambda dtype: pd.api.types.is_numeric_dtype(dtype)).all():
        raise ValueError("❌ X contiene valores no numéricos. Revisa el preprocesamiento.")
    
    try:
        print("📥 Cargando datos procesados...")
        X = pd.read_csv('data/processed/X_processed.csv')
        y = pd.read_csv('data/processed/y.csv').squeeze()

        print(f"📊 Dimensiones de X: {X.shape}, y: {y.shape}")

        # Validar que todos los valores sean numéricos
        if not X.apply(lambda col: pd.to_numeric(col, errors='coerce')).notnull().all().all():
            raise ValueError("❌ Algunas columnas en X no son numéricas. Revisa el preprocesamiento.")

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
        print("📊 Informe de clasificación (Random Forest):")
        print(classification_report(y_test, y_pred_rf))

        # Entrenar XGBoost con Bayesian Search
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
        print("📊 Informe de clasificación (XGBoost):")
        print(classification_report(y_test, y_pred_xgb))

        # Validación cruzada
        print("🔁 Validación cruzada (F1-Score):")
        scores = cross_val_score(bayes_search.best_estimator_, X_train, y_train, cv=5, scoring='f1')
        print(f"📊 F1-Score: {scores.mean():.2f} ± {scores.std():.2f}")

        # Guardar mejor modelo
        best_model = bayes_search
        joblib.dump(best_model, MODEL_PATH)
        print(f"💾 Modelo guardado en {MODEL_PATH}")

    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {str(e)}")
        raise

if __name__ == "__main__":
    train_models()