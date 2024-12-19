import os

# Dossier pour stocker les encodages faciaux
ENCODED_DIR = 'encodings'
META_FILE = os.path.join(ENCODED_DIR, 'metadata.json')

# Créer le dossier s'il n'existe pas
if not os.path.exists(ENCODED_DIR):
    os.makedirs(ENCODED_DIR)

