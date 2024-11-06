import face_recognition
import os
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from PIL import Image
import cv2

# Dossier pour stocker les visages encodés
ENCODED_DIR = 'encodings'

if not os.path.exists(ENCODED_DIR):
    os.makedirs(ENCODED_DIR)


def load_encoded_faces():
    """
    Charger tous les visages encodés existants depuis le dossier ENCODED_DIR.
    """
    encoded_faces = {}
    for file_name in os.listdir(ENCODED_DIR):
        if file_name.endswith(".txt"):
            name = file_name.split(".txt")[0]
            with open(os.path.join(ENCODED_DIR, file_name), 'r') as file:
                encoding = np.array(eval(file.read()))
                encoded_faces[name] = encoding
    return encoded_faces


def face_exists(new_face_encoding, tolerance=0.6):
    """
    Vérifier si le visage encodé existe déjà dans le dossier ENCODED_DIR.
    """
    encoded_faces = load_encoded_faces()

    for name, encoding in encoded_faces.items():
        # Comparer le nouvel encodage avec chaque encodage existant
        matches = face_recognition.compare_faces([encoding], new_face_encoding, tolerance=tolerance)
        if matches[0]:  # Si le visage correspond
            print(f"Le visage correspond à {name}.")
            return True

    print("Aucune correspondance trouvée.")
    return False


def encode_face_from_image(image_path):
    """
    Encoder un visage à partir d'une image.
    """
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    if face_encodings:
        return face_encodings[0]
    else:
        print("Aucun visage détecté dans l'image.")
        return None


def import_and_check_face():
    """
    Importer une image, encoder le visage et vérifier s'il existe déjà.
    """
    Tk().withdraw()
    image_path = askopenfilename(title="Sélectionnez une image", filetypes=[("Images", "*.jpg;*.jpeg;*.png")])

    if not image_path:
        print("Aucune image sélectionnée.")
        return

    # Encoder le visage
    new_face_encoding = encode_face_from_image(image_path)

    if new_face_encoding is not None:
        # Vérifier si le visage existe déjà
        face_exists(new_face_encoding)


if __name__ == "__main__":
    import_and_check_face()
