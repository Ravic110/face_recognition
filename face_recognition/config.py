import os
from pathlib import Path

# Dossier pour stocker les encodages faciaux (chemin absolu)
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ENCODED_DIR = BASE_DIR / "encodings"
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

# Creer le dossier s'il n'existe pas
os.makedirs(ENCODED_DIR, exist_ok=True)
