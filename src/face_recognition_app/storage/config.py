import os
from pathlib import Path

# Racine du projet (4 niveaux au-dessus de ce fichier)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# --- Tolérances pour la reconnaissance faciale ---
# Seuil de distance pour reconnaître un visage (webcam / interface image)
FACE_RECOGNITION_THRESHOLD = 0.5
# Tolérance pour la détection de doublons (utils.py)
DUPLICATE_TOLERANCE = 0.6
# Tolérance pour le regroupement de visages issus d'une vidéo
VIDEO_FACE_TOLERANCE = 0.5

DEFAULT_ENCODED_DIR = PROJECT_ROOT / "encodings"
LEGACY_ENCODED_DIR = Path.cwd() / "encodings"

def _has_json_files(path):
    if not path.exists():
        return False
    return any(p.suffix == ".json" for p in path.glob("*.json"))


# Si l'ancien dossier existe et que le nouveau est vide, on reutilise l'ancien
if LEGACY_ENCODED_DIR.exists() and not _has_json_files(DEFAULT_ENCODED_DIR):
    ENCODED_DIR = str(LEGACY_ENCODED_DIR)
else:
    ENCODED_DIR = str(DEFAULT_ENCODED_DIR)

META_FILE = str(Path(ENCODED_DIR) / "metadata.json")

# Fichier de configuration des caméras (surveillance multi-sources)
CAMERAS_FILE = PROJECT_ROOT / "cameras.json"

# Dossier des événements de surveillance (SQLite uniquement)
# events.db est utilisé pour le stockage des événements

# Créer les dossiers s'ils n'existent pas
os.makedirs(ENCODED_DIR, exist_ok=True)
