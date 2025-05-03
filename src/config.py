from pathlib import Path
import os

# Configuración de rutas
PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA = DATA_DIR / "raw" / "data.csv"
MODELS_DIR = PROJECT_ROOT / "models"

# Mapeo especial de columnas
COLUMN_MAP = {
    'daytime_evening_attendance_': 'attendance_schedule',
    'curricular_units_1st_sem_grade': 'first_sem_grade'
}

# Crear directorios necesarios
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)