import cv2
import face_recognition
import json
from datetime import datetime
import hashlib
import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import logging
from config import ENCODED_DIR, META_FILE
from encodings_store import save_face_encoding as store_save_face_encoding

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@dataclass
class FaceData:
    """Classe pour stocker les données d'un visage"""
    name: str
    encoding: np.ndarray
    timestamp: str
    unique_id: str

class SecureFaceEncoder:
    def __init__(self, encoded_dir: str = ENCODED_DIR, meta_file: str = META_FILE):
        """
        Initialise l'encodeur de visages avec les chemins des fichiers nécessaires.

        Args:
            encoded_dir: Chemin du dossier pour stocker les encodages
            meta_file: Chemin du fichier de métadonnées
        """
        self.encoded_dir = Path(encoded_dir)
        self.meta_file = Path(meta_file)
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Crée les répertoires nécessaires s'ils n'existent pas."""
        self.encoded_dir.mkdir(parents=True, exist_ok=True)

    def _generate_unique_id(self, name: str, timestamp: str) -> str:
        """Génère un identifiant unique pour un encodage."""
        return hashlib.sha256(f"{name}{timestamp}".encode()).hexdigest()[:12]

    def save_face_encoding(self, name: str, frame: np.ndarray, selected_face_idx: int) -> bool:
        """
        Sauvegarde l'encodage d'un visage.

        Args:
            name: Nom de la personne
            frame: Image contenant le visage
            selected_face_idx: Index du visage selectionne

        Returns:
            bool: True si la sauvegarde est reussie, False sinon
        """
        try:
            rgb_frame = self._prepare_frame(frame)
            face_locations = face_recognition.face_locations(rgb_frame)

            if not self._validate_face_detection(face_locations, selected_face_idx):
                return False

            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            store_save_face_encoding(name, face_encodings[selected_face_idx])

            logging.info(f"Encodage pour {name} sauvegarde avec succes.")
            return True

        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde de l'encodage: {str(e)}")
            return False

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        """Prépare l'image pour la détection de visage."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = np.ascontiguousarray(rgb_frame)
        return rgb_frame.astype(np.uint8)

    def _validate_face_detection(self, face_locations: list, selected_face_idx: int) -> bool:
        """Valide la détection du visage et l'index sélectionné."""
        if not face_locations:
            logging.warning("Aucun visage détecté.")
            return False
        if selected_face_idx >= len(face_locations):
            logging.warning("Index de visage invalide.")
            return False
        return True

    def _create_face_data(self, name: str, encoding: np.ndarray) -> FaceData:
        """Crée un objet FaceData avec les informations du visage."""
        timestamp = datetime.now().isoformat()
        unique_id = self._generate_unique_id(name, timestamp)
        return FaceData(name=name, encoding=encoding, timestamp=timestamp, unique_id=unique_id)
    def _load_metadata(self) -> Dict[str, Any]:
        """Charge les métadonnées existantes ou retourne un dictionnaire vide."""
        if self.meta_file.exists():
            with open(self.meta_file, 'r') as f:
                return json.load(f)
        return {}

    def capture_face(self) -> Tuple[Optional[np.ndarray], Optional[str], Optional[int]]:
        """Capture un visage via la webcam."""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Impossible d'accéder à la webcam")

            selected_face_idx = 0
            return self._process_video_capture(cap, selected_face_idx)

        except Exception as e:
            logging.error(f"Erreur lors de la capture: {str(e)}")
            return None, None, None

        finally:
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()

    def _process_video_capture(self, cap: cv2.VideoCapture, selected_face_idx: int) -> Tuple[
        Optional[np.ndarray], Optional[str], Optional[int]]:
        """Traite la capture vidéo et gère l'interface utilisateur."""
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("Erreur lors de la capture d'image")
                return None, None, None

            frame = self._convert_frame_colors(frame)
            face_locations = self._detect_faces(frame)
            frame_with_info = self._draw_interface(frame, face_locations, selected_face_idx)

            cv2.imshow("Capture de visage", frame_with_info)

            action = self._handle_key_press()
            if action == "save" and face_locations:
                return self._finalize_capture(frame, face_locations, selected_face_idx)
            elif action == "next" and face_locations:
                selected_face_idx = (selected_face_idx + 1) % len(face_locations)
            elif action == "quit":
                logging.info("Capture annulée par l'utilisateur")
                return None, None, None

    def _convert_frame_colors(self, frame: np.ndarray) -> np.ndarray:
        """Convertit les couleurs de l'image pour l'affichage."""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def _detect_faces(self, frame: np.ndarray) -> list:
        """Détecte les visages dans l'image."""
        rgb_frame = self._prepare_frame(frame)
        return face_recognition.face_locations(rgb_frame)

    def _draw_interface(self, frame: np.ndarray, face_locations: list, selected_face_idx: int) -> np.ndarray:
        """Dessine l'interface utilisateur sur l'image."""
        frame_with_info = frame.copy()

        # Instructions
        self._draw_instructions(frame_with_info, len(face_locations), selected_face_idx)

        # Rectangles de détection
        self._draw_face_rectangles(frame_with_info, face_locations, selected_face_idx)

        return frame_with_info

    def _draw_instructions(self, frame: np.ndarray, num_faces: int, selected_face_idx: int) -> None:
        """Dessine les instructions sur l'image."""
        cv2.putText(frame, "Appuyez sur 's' pour sauvegarder",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Appuyez sur 'q' pour quitter",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if num_faces > 1:
            cv2.putText(frame, "Appuyez sur 'n' pour changer de visage",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Visage sélectionné: {selected_face_idx + 1}/{num_faces}",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def _draw_face_rectangles(self, frame: np.ndarray, face_locations: list, selected_face_idx: int) -> None:
        """Dessine les rectangles autour des visages détectés."""
        for idx, (top, right, bottom, left) in enumerate(face_locations):
            color = (0, 255, 0) if idx == selected_face_idx else (255, 0, 0)
            thickness = 2 if idx == selected_face_idx else 1
            cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)
            cv2.putText(frame, f"#{idx + 1}",
                        (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness)

    def _handle_key_press(self) -> Optional[str]:
        """Gère les touches pressées par l'utilisateur."""
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            return "save"
        elif key == ord('n'):
            return "next"
        elif key == ord('q'):
            return "quit"
        return None

    def _finalize_capture(self, frame: np.ndarray, face_locations: list, selected_face_idx: int) -> Tuple[
        np.ndarray, str, int]:
        """Finalise la capture en demandant le nom."""
        preview_frame = self._create_preview(frame, face_locations, selected_face_idx)
        cv2.imshow("Prévisualisation", preview_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        name = input("Entrez le nom de la personne : ")
        return frame, name, selected_face_idx

    def _create_preview(self, frame: np.ndarray, face_locations: list, selected_face_idx: int) -> np.ndarray:
        """Crée l'image de prévisualisation."""
        preview = frame.copy()
        if face_locations:
            top, right, bottom, left = face_locations[selected_face_idx]
            cv2.rectangle(preview, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(preview, "Visage sélectionné - Appuyez sur une touche pour continuer",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return preview

if __name__ == "__main__":
    try:
        encoder = SecureFaceEncoder()
        frame, name, face_idx = encoder.capture_face()
        if frame is not None and name:
            encoder.save_face_encoding(name, frame, face_idx)
    except Exception as e:
        logging.error(f"Erreur lors de l'exécution: {str(e)}")
