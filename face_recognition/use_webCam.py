import cv2
import face_recognition
import os
import json
from datetime import datetime
import hashlib
import numpy as np
from config import ENCODED_DIR, META_FILE


class SecureFaceEncoder:
    def __init__(self):
        self.ENCODED_DIR = ENCODED_DIR
        self.META_FILE = META_FILE

        if not os.path.exists(self.ENCODED_DIR):
            os.makedirs(self.ENCODED_DIR)

    def save_face_encoding(self, name, frame, selected_face_idx):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = np.ascontiguousarray(rgb_frame)

        if rgb_frame.dtype != np.uint8:
            rgb_frame = rgb_frame.astype(np.uint8)

        face_locations = face_recognition.face_locations(rgb_frame)
        if not face_locations:
            print("Aucun visage détecté. Veuillez réessayer.")
            return False

        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        if not face_encodings or selected_face_idx >= len(face_encodings):
            print("Impossible d'encoder le visage sélectionné. Veuillez réessayer.")
            return False

        timestamp = datetime.now().isoformat()
        unique_id = hashlib.sha256(f"{name}{timestamp}".encode()).hexdigest()[:12]

        encoding_data = {
            'name': name,
            'encoding': face_encodings[selected_face_idx].tolist(),
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

    def capture_face(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Erreur: Impossible d'accéder à la webcam")
            return

        selected_face_idx = 0
        captured_frame = None
        face_locations = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Erreur lors de la capture d'image")
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

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

                if len(face_locations) > 1:
                    cv2.putText(frame_with_info, "Appuyez sur 'n' pour changer de visage",
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame_with_info, f"Visage sélectionné: {selected_face_idx + 1}/{len(face_locations)}",
                                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Affichage des rectangles de détection
                for idx, (top, right, bottom, left) in enumerate(face_locations):
                    color = (0, 255, 0) if idx == selected_face_idx else (255, 0, 0)
                    thickness = 2 if idx == selected_face_idx else 1
                    cv2.rectangle(frame_with_info, (left, top), (right, bottom), color, thickness)
                    cv2.putText(frame_with_info, f"#{idx + 1}",
                                (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness)

                cv2.imshow("Capture de visage", frame_with_info)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('s') and face_locations:
                    captured_frame = frame.copy()
                    break
                elif key == ord('n') and face_locations:
                    selected_face_idx = (selected_face_idx + 1) % len(face_locations)
                elif key == ord('q'):
                    print("Capture annulée par l'utilisateur")
                    return None, None

        finally:
            cap.release()
            cv2.destroyAllWindows()

        if captured_frame is not None:
            # Afficher l'image capturée avec le visage sélectionné
            preview_frame = captured_frame.copy()
            if face_locations:
                top, right, bottom, left = face_locations[selected_face_idx]
                cv2.rectangle(preview_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(preview_frame, "Visage sélectionné - Appuyez sur une touche pour continuer",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Prévisualisation", preview_frame)
            print("Prévisualisation affichée. Appuyez sur une touche pour continuer...")
            cv2.waitKey(0)  # Attend une entrée de l'utilisateur
            cv2.destroyAllWindows()

            # Demander le nom après la capture
            name = input("Entrez le nom de la personne : ")
            return captured_frame, name, selected_face_idx

        return None, None, None


if __name__ == "__main__":
    encoder = SecureFaceEncoder()
    frame, name, face_idx = encoder.capture_face()
    if frame is not None and name:
        encoder.save_face_encoding(name, frame, face_idx)