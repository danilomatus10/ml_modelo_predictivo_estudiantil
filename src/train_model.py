# src/train_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from skopt import BayesSearchCV
from config import MODEL_PATH

def train_models():
    X = pd.read_csv('data/processed/X_processed.csv')
    y = pd.read_csv('data/processed/y.csv').squeeze()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Random Forest con GridSearch
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    grid_rf = GridSearchCV(RandomForestClassifier(), rf_params, cv=5, scoring='f1')
    grid_rf.fit(X_train, y_train)
    
    # XGBoost con Bayesian Optimization
    xgb_params = {
        'n_estimators': (50, 200),
        'max_depth': (3, 10),
        'learning_rate': (0.01, 0.3, 'log-uniform')
    }
    
    bayes_xgb = BayesSearchCV(XGBClassifier(use_label_encoder=False), xgb_params, n_iter=30, cv=5, scoring='f1')
    bayes_xgb.fit(X_train, y_train)
    
    # Comparar modelos
    models = {
        'Random Forest': grid_rf,
        'XGBoost': bayes_xgb
    }
    
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        results.append((name, f1, auc))
        print(f"{name} - F1: {f1:.4f}, AUC: {auc:.4f}")
    
    # Guardar mejor modelo
    best_model = max(models.values(), key=lambda x: x.best_score_)
    joblib.dump(best_model, MODEL_PATH)
    print(f"Mejor modelo guardado en {MODEL_PATH}")

if __name__ == "__main__":
    train_models()