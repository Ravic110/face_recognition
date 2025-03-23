import os
import json
import cv2, base64
import numpy as np
from datetime import datetime
from face_recognition import compare_faces, face_distance
from config import ENCODED_DIR, META_FILE


def load_existing_encodings():
    encodings = []
    if not os.path.exists(ENCODED_DIR):
        return encodings

    for filename in os.listdir(ENCODED_DIR):
        filepath = os.path.join(ENCODED_DIR, filename)
        try:
            if filename.endswith(".json") and 'temp' not in filename:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    if validate_encoding(data):
                        encodings.append({
                            'name': data['name'],
                            'encoding': np.array(data['encoding'])
                        })
                    else:
                        os.remove(filepath)
        except Exception as e:
            os.remove(filepath)
    return encodings


def save_face_encoding(name, encoding, image):
    timestamp = datetime.now().isoformat()
    unique_id = f"{name}_{timestamp}".encode('utf-8').hex()[:12]

    # Conversion de l'image
    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    data = {
        'name': name,
        'encoding': encoding.tolist(),
        'image':  image_base64,
        'timestamp': timestamp
    }

    os.makedirs(ENCODED_DIR, exist_ok=True)
    with open(os.path.join(ENCODED_DIR, f"{unique_id}.json"), 'w') as f:
        json.dump(data, f, indent=4)

    update_metadata(unique_id, name)


def is_duplicate(encoding, existing_encodings, tolerance=0.6):
    # Première passe rapide
    encodings_list = [e['encoding'] for e in existing_encodings]
    matches = compare_faces(encodings_list, encoding, tolerance + 0.1)

    # Deuxième passe précise
    for i, match in enumerate(matches):
        if match:
            distance = face_distance([existing_encodings[i]['encoding']], encoding)[0]
            if distance < tolerance:
                return existing_encodings[i]['name']
    return None


def validate_encoding(data):
    required = ['name', 'encoding', 'timestamp']
    return all(key in data for key in required) and len(data['encoding']) == 128


def update_metadata(uid, name):
    metadata = {}
    if os.path.exists(META_FILE):
        with open(META_FILE, 'r') as f:
            metadata = json.load(f)

    metadata[uid] = {
        'name': name,
        'date': datetime.now().isoformat()
    }

    with open(META_FILE, 'w') as f:
        json.dump(metadata, f, indent=4)


def delete_encoding(name):
    print(f"Tentative de suppression de l'encoding : {name}")
    for filename in os.listdir(ENCODED_DIR):
        if filename.endswith(".json") and name in filename:
            filepath = os.path.join(ENCODED_DIR, filename)
            print(f"Suppression du fichier : {filepath}")
            os.remove(filepath)

    if os.path.exists(META_FILE):
        with open(META_FILE, 'r') as f:
            metadata = json.load(f)

        metadata = {k: v for k, v in metadata.items() if v['name'] != name}

        with open(META_FILE, 'w') as f:
            json.dump(metadata, f)
        print(f"Metadata mise à jour pour supprimer : {name}")