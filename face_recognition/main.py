import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES
import cv2
import face_recognition
import os
import numpy as np

# Dossier pour stocker les encodages des visages
ENCODED_DIR = 'encodings'


def load_face_encodings():
    """
    Charger tous les encodages des visages enregistrés.
    """
    known_face_encodings = []
    known_face_names = []

    for file_name in os.listdir(ENCODED_DIR):
        if file_name.endswith('.txt'):
            with open(f"{ENCODED_DIR}/{file_name}", 'r') as file:
                encoding_str = file.read()
                face_encoding = np.array(eval(encoding_str))
                known_face_encodings.append(face_encoding)
                known_face_names.append(file_name[:-4])  # Retirer ".txt"

    return known_face_encodings, known_face_names


def recognize_faces_in_image(image_path):
    """
    Reconnaître des visages dans une image donnée.
    """
    known_face_encodings, known_face_names = load_face_encodings()

    # Charger l'image
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Détection et encodage des visages dans l'image
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Inconnu"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Afficher le nom et rectangle autour du visage
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Afficher l'image avec les visages reconnus
    cv2.imshow("Résultats de la reconnaissance", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def on_drop(event):
    """
    Fonction appelée lors du glisser-déposer d'une image.
    """
    file_path = event.data.strip('{}')  # Supprimer les accolades de TkinterDND
    recognize_faces_in_image(file_path)


# Interface graphique (GUI) avec drag-and-drop
class FaceRecognitionApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()

        self.title("Système de reconnaissance faciale - Drag and Drop")

        # Label pour la zone de glisser-déposer
        self.drop_area_label = tk.Label(self, text="Glisser et déposer une image ici", width=40, height=10,
                                        bg="lightgray")
        self.drop_area_label.pack(pady=20)

        # Zone qui accepte le drop de fichiers
        self.drop_area_label.drop_target_register(DND_FILES)
        self.drop_area_label.dnd_bind('<<Drop>>', on_drop)

        # Bouton pour quitter l'application
        self.quit_button = tk.Button(self, text="Quitter", command=self.quit)
        self.quit_button.pack(pady=20)


if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.mainloop()
