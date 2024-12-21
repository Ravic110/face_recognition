import face_recognition
import os
import json
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from config import ENCODED_DIR, META_FILE

if not os.path.exists(ENCODED_DIR):
    os.makedirs(ENCODED_DIR)


def load_metadata():
    if os.path.exists(META_FILE):
        with open(META_FILE, 'r') as file:
            return json.load(file)
    return {}


def load_encoded_faces():
    """
    Charger tous les encodages de visages existants depuis les fichiers JSON dans ENCODED_DIR,
    en ignorant les fichiers de métadonnées.
    """
    encoded_faces = {}
    print("Chargement des encodages...")
    for file_name in os.listdir(ENCODED_DIR):
        if file_name.endswith(".json") and file_name != os.path.basename(META_FILE):
            file_path = os.path.join(ENCODED_DIR, file_name)
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    if 'name' in data and 'encoding' in data:
                        encoded_faces[data['name']] = np.array(data['encoding'])
                    else:
                        print(f"Fichier JSON mal formé ou incomplet - {file_path}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Erreur lors du chargement du fichier {file_name} : {e}")
    print(f"Total des encodages chargés : {len(encoded_faces)}")
    return encoded_faces


def encode_face_from_image(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    if not face_encodings:
        print("Erreur : aucun visage détecté ou image de mauvaise qualité.")
        return None

    if len(face_encodings) > 1:
        print("Attention : plusieurs visages détectés. Le premier visage sera utilisé.")
    return face_encodings[0]


def face_exists(new_face_encoding, tolerance=0.6):
    """
    Vérifier si le visage existe déjà dans les encodages.
    """
    encoded_faces = load_encoded_faces()
    for name, encoding in encoded_faces.items():
        matches = face_recognition.compare_faces([encoding], new_face_encoding, tolerance=tolerance)
        if matches[0]:
            print(f"Le visage correspond à {name}.")
            return True

    print("Aucune correspondance trouvée.")
    return False


def import_and_check_face():
    """
    Importer une image, détecter un visage et vérifier s'il existe déjà.
    """
    Tk().withdraw()
    image_path = askopenfilename(title="Sélectionnez une image", filetypes=[("Images", "*.jpg;*.jpeg;*.png")])

    if not image_path:
        print("Aucune image sélectionnée.")
        return

    new_face_encoding = encode_face_from_image(image_path)
    if new_face_encoding is not None:
        print("\nOptions de tolérance :")
        print("1. Faible (0.4)")
        print("2. Moyenne (0.6)")
        print("3. Élevée (0.8)")
        tolerance_choice = input("Choisissez un niveau de tolérance (1/2/3) : ")

        tolerance_map = {'1': 0.4, '2': 0.6, '3': 0.8}
        tolerance = tolerance_map.get(tolerance_choice, 0.6)

        face_exists(new_face_encoding, tolerance)


if __name__ == "__main__":
    import_and_check_face()
