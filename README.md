# Modelo Predictivo de Deserción Estudiantil

🧰 Instalación
# 1. Clonar el repositorio
git clone https://github.com/danilomatus10/ml_modelo_predictivo_estudiantil.git
cd ml_modelo_predictivo_estudiantil

# 2. Crear y activar el entorno
conda env create -f environment.yml
conda activate ml_modelo_predictivo_estudiantil

# 3. Instalar dependencias adicionales (si es necesario)
pip install imbalanced-learn shap

📁 Estructura del Proyecto

ml_modelo_predictivo_estudiantil/
├── data/
│   ├── raw/                   # Datos originales (data.csv)
│   └── processed/             # Datos limpios (cleaned_data.csv)
│
├── notebooks/                 # Análisis y visualizaciones
│   ├── 01_EDA.ipynb           # Análisis Exploratorio de Datos
│   └── 03_Resultados.ipynb    # Resultados del modelo y métricas
│
├── src/                       # Código fuente
│   ├── config.py              # Configuración de rutas
│   ├── data_loader.py         # Carga y limpieza de datos
│   ├── preprocessing.py       # Preprocesamiento y Feature Engineering
│   └── train_model.py         # Entrenamiento del modelo
│
├── models/                    # Modelos entrenados (best_model.pkl)
├── reports/                   # Informes generados (HTML/PDF)
└── tests/                     # Pruebas unitarias

▶️ Ejecución del Proyecto

# 1. Coloca el archivo data.csv en data/raw/
# 2. Activa el entorno
conda activate ml_modelo_predictivo_estudiantil

# 3. Cargar y limpiar datos
python src/data_loader.py

# 4. Preprocesamiento (genera X_processed.csv y y.csv)
python src/preprocessing.py

# 5. Análisis Exploratorio (opcional)
jupyter notebook notebooks/01_EDA.ipynb

# 6. Entrenamiento del modelo
python src/train_model.py

# 7. Generar informe del EDA
mkdir -p reports
jupyter nbconvert --to html notebooks/01_EDA.ipynb --output reports/EDA_Report.html

# 8. Ejecutar pruebas unitarias
pytest tests/

📊 Resultados Clave 
Métricas del Modelo Final (XGBoost Optimizado): 

F1-Score
	
0.83
AUC-ROC
	
0.91
Accuracy
	
0.87

Características Más Relevantes: 

    Nota de admisión (Admission grade)
    Tasa de aprobación primer semestre (Approval_rate_1st_sem)
    Índice económico (Economic_index)
     
📋 Autores 

    Hubert Gutiérrez   
    Danulo Matus   
    Emilley Roque 


🛠️ Resolución de Problemas Comunes 
❌ Error: "could not convert string to float" 

Causa : Archivos CSV procesados con matrices dispersas (formato incorrecto).
Solución :   

    1. Asegúrate de que preprocessing.py incluya:
    X_resampled = X_resampled.toarray()  # Convierte a matriz densa

    Elimina archivos antiguos antes de regenerar:
     

rm data/processed/X_processed.csv
rm data/processed/y.csv
python src/preprocessing.py

 
❌ Error: "No module named 'notebook.services'" 

Causa : Falta la librería notebook para nbconvert.
Solución :   

1 pip install notebook
 
 
📁 Notas Importantes 

    Datos confidenciales : Si data.csv contiene información sensible, no lo subas a GitHub . Envíalo al docente por correo.
    Reproducibilidad : Todos los pasos deben ejecutarse en orden para garantizar consistencia.
    Interpretación del modelo : Usa SHAP para explicar predicciones individuales.
     

