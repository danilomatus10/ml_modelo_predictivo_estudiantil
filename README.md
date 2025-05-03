# Modelo Predictivo de Deserción Estudiantil

## Instalación
```bash
conda env create -f environment.yml
conda activate ml_proyecto_modelo_estudiantil

#Ejecución

# Colocar el archivo data.csv en data/raw/
# Ejecutar análisis exploratorio:
jupyter notebook notebooks/01_EDA.ipynb

#Entrenar modelo:
python src/train_model.py

#Estructura
data/       # Datos crudos y procesados
notebooks/  # Análisis exploratorio
src/        # Código fuente
models/     # Modelos entrenados
tests/      # Pruebas unitarias


---

#Autores:

[Hubert Gutiérrez]
[Danulo Matus]
[Emllely Roque]

### **Ejecución del Proyecto**
```bash
# Desde la raíz del proyecto
conda activate ml_proyecto_modelo_estudiantil
pytest tests/  # Ejecutar pruebas
python src/train_model.py  # Entrenar modelo
