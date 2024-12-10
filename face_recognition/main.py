import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES
import cv2
import face_recognition
import os
import json
import numpy as np
from PIL import Image, ImageTk

# Dossier pour stocker les encodages des visages
ENCODED_DIR = 'encodings'


def save_face_encoding(name, encoding):
    """
    Enregistrer un encodage de visage dans un fichier JSON.
    """
    if not os.path.exists(ENCODED_DIR):
        os.makedirs(ENCODED_DIR)

    file_path = os.path.join(ENCODED_DIR, f"{name}.json")
    with open(file_path, 'w') as file:
        json.dump({"name": name, "encoding": encoding.tolist()}, file)
    print(f"Encodage enregistré pour : {name}")


def load_face_encodings():
    """
    Charger tous les encodages des visages enregistrés.
    """
    known_face_encodings = []
    known_face_names = []

    # Vérifier l'existence du dossier d'encodage
    if not os.path.exists(ENCODED_DIR):
        os.makedirs(ENCODED_DIR)
        print(f"Le dossier '{ENCODED_DIR}' a été créé car il n'existait pas.")
        return known_face_encodings, known_face_names

    # Charger les fichiers JSON d'encodage
    for file_name in os.listdir(ENCODED_DIR):
        if file_name.endswith('.json'):
            with open(f"{ENCODED_DIR}/{file_name}", 'r') as file:
                encoding_data = json.load(file)
                face_encoding = np.array(encoding_data['encoding'])
                known_face_encodings.append(face_encoding)
                known_face_names.append(encoding_data['name'])

    return known_face_encodings, known_face_names

def detect_and_register_faces(image_path, display_message_callback):
    """
    Détecter les visages dans une image et permettre leur enregistrement.
    """
    try:
        # Chargement robuste de l'image
        image = cv2.imread(image_path)
        if image is None:
            display_message_callback("Impossible de charger l'image avec OpenCV. Tentative avec PIL...")
            pil_image = Image.open(image_path)
            image = np.array(pil_image)

        # Vérifiez si l'image est déjà en 8 bits
        if image.ndim == 2:  # Image en niveaux de gris
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 3:  # Image RGB
            rgb_image = image
        else:
            raise ValueError("Format d'image non pris en charge (pas 8 bits ou pas RGB).")

        # Détection des visages
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        if not face_locations:
            display_message_callback("Aucun visage détecté.")
            return

        for i, (top, right, bottom, left) in enumerate(face_locations):
            # Dessiner un rectangle autour du visage
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

            # Demander un nom pour le visage
            name = input(f"Entrez un nom pour le visage détecté {i + 1}: ").strip()

            if name:
                save_face_encoding(name, face_encodings[i])

        # Afficher l'image avec les rectangles autour des visages détectés
        display_message_callback(f"{len(face_locations)} visage(s) détecté(s) et enregistré(s).")
        cv2.imshow("Visages détectés", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        display_message_callback(f"Erreur : {e}")



# Interface graphique (GUI) avec drag-and-drop
class FaceRecognitionApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()

        self.title("Système de reconnaissance faciale - Détection et Enregistrement")
        self.geometry("600x600")

        # Zone pour afficher l'image déposée
        self.image_label = tk.Label(self, text="Aucun aperçu disponible", bg="lightgray", width=60, height=20)
        self.image_label.pack(pady=10)

        # Label pour la zone de message
        self.message_label = tk.Label(self, text="", fg="blue")
        self.message_label.pack(pady=10)

        # Label pour la zone de glisser-déposer
        self.drop_area_label = tk.Label(self, text="Glisser et déposer une image ici", width=40, height=5,
                                        bg="lightblue")
        self.drop_area_label.pack(pady=10)

        # Zone qui accepte le drop de fichiers
        self.drop_area_label.drop_target_register(DND_FILES)
        self.drop_area_label.dnd_bind('<<Drop>>', self.on_drop)

        # Bouton pour quitter l'application
        self.quit_button = tk.Button(self, text="Quitter", command=self.quit)
        self.quit_button.pack(pady=10)

    def display_message(self, message):
        """
        Mettre à jour le label de message.
        """
        self.message_label.config(text=message)

    def display_image(self, image_path):
        """
        Afficher l'aperçu de l'image.
        """
        image = Image.open(image_path)
        image.thumbnail((400, 300))  # Redimensionner pour l'affichage
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo

    def on_drop(self, event):
        """
        Fonction appelée lors du glisser-déposer d'une image.
        """
        file_path = event.data.strip('{}')  # Supprimer les accolades de TkinterDND
        self.display_message("Chargement de l'image...")
        self.display_image(file_path)

        # Détecter et enregistrer les visages
        detect_and_register_faces(file_path, self.display_message)


if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.mainloop()
