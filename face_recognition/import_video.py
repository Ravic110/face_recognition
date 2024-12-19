import cv2
import face_recognition
import os
import json
from datetime import datetime
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from config import ENCODED_DIR,META_FILE
import numpy as np

# Dossier et fichier de métadonnées partagés

if not os.path.exists(ENCODED_DIR):
    os.makedirs(ENCODED_DIR)

def save_face_encoding(name, face_encoding):
    """
    Enregistrer l'encodage du visage détecté dans un fichier et mettre à jour les métadonnées.
    """
    # Génération d'un identifiant unique
    timestamp = datetime.now().isoformat()
    unique_id = f"{hash(name + timestamp) & 0xFFFFFFFF:08x}"

    # Structure des données
    encoding_data = {
        'name': name,
        'encoding': face_encoding.tolist(),
        'timestamp': timestamp
    }

    # Sauvegarder l'encodage
    file_path = os.path.join(ENCODED_DIR, f"{unique_id}.json")
    with open(file_path, 'w') as file:
        json.dump(encoding_data, file)

    # Mise à jour des métadonnées
    update_metadata(unique_id, name)
    print(f"Encodage pour {name} sauvegardé avec succès.")

def update_metadata(unique_id, name):
    """
    Mettre à jour le fichier de métadonnées avec les nouvelles informations.
    """
    metadata = {}
    if os.path.exists(META_FILE):
        with open(META_FILE, 'r') as file:
            metadata = json.load(file)

    metadata[unique_id] = {
        'name': name,
        'date_creation': datetime.now().isoformat()
    }

    with open(META_FILE, 'w') as file:
        json.dump(metadata, file, indent=4)

def process_video(video_path, name):
    """
    Analyser la vidéo pour détecter les visages dans chaque frame et enregistrer les encodages.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Erreur lors de l'ouverture de la vidéo.")
        return

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Processer une image sur 10 pour éviter les doublons inutiles
        if frame_count % 10 != 0:
            continue

        # Conversion de l'image en format RGB pour face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Détection des visages
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Dessiner des rectangles autour des visages détectés
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Sauvegarder les encodages des visages détectés
        for face_encoding in face_encodings:
            save_face_encoding(name, face_encoding)

        # Afficher le résultat
        cv2.imshow("Détection de visages", frame)

        # Quitter la boucle si la touche 'q' est appuyée
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def import_video():
    """
    Ouvrir une boîte de dialogue pour sélectionner une vidéo et traiter les visages détectés.
    """
    Tk().withdraw()  # Masquer la fenêtre Tkinter par défaut
    video_path = askopenfilename(title="Sélectionnez une vidéo", filetypes=[("Vidéos", "*.mp4;*.avi;*.mkv")])

    if not video_path:
        print("Aucune vidéo sélectionnée.")
        return

    name = input("Entrez le nom de la personne : ")

    # Traiter la vidéo pour la détection et l'encodage des visages
    process_video(video_path, name)

if __name__ == "__main__":
    print("Détection de visages dans une vidéo")
    import_video()
