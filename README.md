# Modelo Predictivo de Deserción Estudiantil 🎓

## 📁 Archivos generados
| Archivo | Propósito | Justificación |
|--------|----------|---------------|
| `cleaned_data.csv` | Datos limpios y validados | Para evitar repetir limpieza cada vez que se cargan datos |
| `X_processed.csv` | Características procesadas | Listas para entrenamiento (numéricas, escaladas, balanceadas) |
| `y.csv` | Variable objetivo | Etiquetas binarias (0, 1) para clasificación |
| `best_model.pkl` | Modelo entrenado | Reutilización sin reentrenamiento |

# 1. Cargar y limpiar datos
python src/data_loader.py

# 2. Preprocesamiento (con conversión a denso)
python src/preprocessing.py

# 3. Entrenamiento con Bayesian Search
python src/train_model.py

# 4. Análisis exploratorio
jupyter notebook notebooks/01_EDA.ipynb

# 5. Análisis de clústeres
jupyter notebook notebooks/02_Clustering.ipynb

# 6. Análisis de resultados
jupyter notebook notebooks/03_Resultados.ipynb

# 7. Generar informe EDA
jupyter nbconvert --to html notebooks/01_EDA.ipynb --output reports/EDA_Report.html

# 8. Ejecutar pruebas unitarias
pytest tests/

## ¿Por qué generamos `X_processed.csv` y `y.csv`?
Estos archivos contienen:
- **`X_processed.csv`**: Características procesadas (escaladas, codificadas y balanceadas)
- **`y.csv`**: Variable objetivo (`Target`) con etiquetas `0` (Dropout) y `1` (Graduate)

### ¿Por qué los separamos?
- **Reproducibilidad**: Permite reentrenar modelos sin repetir todo el preprocesamiento
- **Eficiencia**: Mejora el rendimiento al cargar solo las características o el objetivo por separado
- **Claridad**: Facilita la interpretación y documentación del proceso

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
│   ├── raw/                   # Datos originales (no versionados)
│   │   └── data.csv         # Archivo CSV original
│   └── processed/             # Datos procesados
│       ├── cleaned_data.csv   # Datos limpios
│       ├── X_processed.csv    # Datos procesados
│       └── y.csv            # Etiquetas
│
├── notebooks/                 # Notebooks Jupyter
│   ├── 01_EDA.ipynb         # Análisis Exploratorio
│   ├── 02_Clustering.ipynb   # Análisis de clústeres
│   └── 03_Resultados.ipynb  # Resultados del modelo
│
├── src/                       # Código fuente
│   ├── config.py              # Configuración de rutas
│   ├── data_loader.py         # Carga de datos
│   ├── preprocessing.py       # Pipelines de preprocesamiento
│   └── train_model.py         # Entrenamiento de modelos
│
├── models/                    # Modelos entrenados
│   └── best_model.pkl        # Modelo serializado
│
├── reports/                   # Informes y visualizaciones
│   ├── EDA_Report.html
│   ├── feature_importance.png
│   ├── clusters_visualization.png
│   ├── confusion_matrix.png
│   └── shap_importance.png
│
├── tests/                     # Pruebas unitarias
│   ├── test_preprocessing.py
│   └── test_data_loading.py
│
├── environment.yml            # Entorno actualizado
└── README.md                  # Documentación del proyecto

▶️ Ejecución del Proyecto

# 1. Cargar y limpiar datos
python src/data_loader.py

# 2. Preprocesamiento (con validación de datos numéricos)
python src/preprocessing.py

# 3. Entrenamiento con Bayesian Search
python src/train_model.py

# 4. Generar informe EDA
jupyter nbconvert --to html notebooks/01_EDA.ipynb --output reports/EDA_Report.html

# 5. Análisis de clústeres
jupyter notebook notebooks/02_Clustering.ipynb

# 6. Análisis de resultados
jupyter notebook notebooks/03_Resultados.ipynb

# 7. Ejecutar pruebas unitarias
pytest tests/


📋 Autores 

    Hubert Gutiérrez   
    Danilo Matus   
    Enllely Roque 


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
     

