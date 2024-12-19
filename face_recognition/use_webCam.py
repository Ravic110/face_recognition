import cv2
import face_recognition
import os
import json
from datetime import datetime
import hashlib
import numpy as np
from config import ENCODED_DIR,META_FILE

class SecureFaceEncoder:
    def __init__(self):
        self.ENCODED_DIR = ENCODED_DIR
        self.META_FILE = META_FILE

        if not os.path.exists(self.ENCODED_DIR):
            os.makedirs(self.ENCODED_DIR)

    def save_face_encoding(self, name, frame):
        # Conversion en RGB avec vérification du type et de la plage de valeurs
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = np.ascontiguousarray(rgb_frame)  # Assure que l'array est contigu en mémoire

        # Vérification du type et de la plage de valeurs
        if rgb_frame.dtype != np.uint8:
            rgb_frame = rgb_frame.astype(np.uint8)

        # Détection et encodage des visages
        face_locations = face_recognition.face_locations(rgb_frame)
        if not face_locations:
            print("Aucun visage détecté. Veuillez réessayer.")
            return False

        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        if not face_encodings:
            print("Impossible d'encoder le visage. Veuillez réessayer.")
            return False

        # Création d'un identifiant unique
        timestamp = datetime.now().isoformat()
        unique_id = hashlib.sha256(f"{name}{timestamp}".encode()).hexdigest()[:12]

        # Sauvegarde de l'encodage
        encoding_data = {
            'name': name,
            'encoding': face_encodings[0].tolist(),
            'timestamp': timestamp
        }

        file_path = os.path.join(self.ENCODED_DIR, f"{unique_id}.json")
        with open(file_path, 'w') as f:
            json.dump(encoding_data, f, indent=4)

        self.update_metadata(unique_id, name)
        print(f"Encodage pour {name} sauvegardé avec succès.")
        return True

    def update_metadata(self, unique_id, name):
        metadata = {}
        if os.path.exists(self.META_FILE):
            with open(self.META_FILE, 'r') as f:
                metadata = json.load(f)

        metadata[unique_id] = {
            'name': name,
            'date_creation': datetime.now().isoformat()
        }

        with open(self.META_FILE, 'w') as f:
            json.dump(metadata, f, indent=4)

    def capture_face(self, name):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Erreur: Impossible d'accéder à la webcam")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Erreur lors de la capture d'image")
                    break

                # Conversion et vérification du format d'image
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Reconversion pour l'affichage

                # Détection des visages pour l'affichage
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame = np.ascontiguousarray(rgb_frame)
                if rgb_frame.dtype != np.uint8:
                    rgb_frame = rgb_frame.astype(np.uint8)

                face_locations = face_recognition.face_locations(rgb_frame)

                frame_with_info = frame.copy()

                # Affichage des instructions
                cv2.putText(frame_with_info, "Appuyez sur 's' pour sauvegarder",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame_with_info, "Appuyez sur 'q' pour quitter",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Affichage du rectangle de détection
                for (top, right, bottom, left) in face_locations:
                    cv2.rectangle(frame_with_info, (left, top), (right, bottom), (0, 255, 0), 1)

                cv2.imshow("Capture de visage", frame_with_info)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    if self.save_face_encoding(name, frame):
                        break
                elif key == ord('q'):
                    print("Capture annulée par l'utilisateur")
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    encoder = SecureFaceEncoder()
    name = input("Entrez le nom de la personne : ")
    encoder.capture_face(name)
