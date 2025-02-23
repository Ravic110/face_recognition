import os
import json
import hashlib
from datetime import datetime
from config import ENCODED_DIR, META_FILE

def load_existing_encodings():
    encodings = []
    if os.path.exists(ENCODED_DIR):
        for filename in os.listdir(ENCODED_DIR):
            # Ignorer le fichier metadata.json
            if filename == "metadata.json":
                continue

            if filename.endswith(".json"):
                try:
                    with open(os.path.join(ENCODED_DIR, filename), 'r') as f:
                        data = json.load(f)
                        # Vérifier que les clés 'name' et 'encoding' existent
                        if 'name' in data and 'encoding' in data:
                            encodings.append({'name': data['name'], 'encoding': data['encoding']})
                        else:
                            print(f"Avertissement : Le fichier {filename} est mal formaté et sera ignoré.")
                except json.JSONDecodeError:
                    print(f"Avertissement : Le fichier {filename} n'est pas un JSON valide et sera ignoré.")
                except Exception as e:
                    print(f"Avertissement : Erreur lors de la lecture du fichier {filename} : {e}")
    return encodings

def is_duplicate(encoding, existing_encodings, tolerance=0.5):
    from face_recognition import compare_faces
    known_encodings = [entry['encoding'] for entry in existing_encodings]
    results = compare_faces(known_encodings, encoding, tolerance)
    if True in results:
        index = results.index(True)
        return existing_encodings[index]['name']
    return None

def save_face_encoding(name, face_encoding):
    timestamp = datetime.now().isoformat()
    unique_id = hashlib.sha256(f"{name}{timestamp}".encode()).hexdigest()[:12]

    encoding_data = {
        'name': name,
        'encoding': face_encoding.tolist(),
        'timestamp': timestamp
    }

    file_path = os.path.join(ENCODED_DIR, f"{unique_id}.json")
    with open(file_path, 'w') as file:
        json.dump(encoding_data, file, indent=4)

    update_metadata(unique_id, name)

def update_metadata(unique_id, name):
    metadata = {}
    if os.path.exists(META_FILE):
        with open(META_FILE, 'r') as f:
            metadata = json.load(f)

    metadata[unique_id] = {
        'name': name,
        'date_creation': datetime.now().isoformat()
    }

    with open(META_FILE, 'w') as f:
        json.dump(metadata, f, indent=4)