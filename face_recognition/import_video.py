from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
import face_recognition
import logging
from collections import defaultdict


@dataclass
class DetectedFace:
    frame_number: int
    location: Tuple[int, int, int, int]
    encoding: np.ndarray
    quality: float
    image: np.ndarray


class FaceCollector:
    def __init__(self):
        self.detected_faces: List[DetectedFace] = []
        self.selected_faces: Dict[str, List[DetectedFace]] = defaultdict(list)

    def collect_faces(self, frame: np.ndarray, frame_number: int) -> List[Tuple[int, int, int, int]]:
        """Détecte et collecte les visages d'une frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for location, encoding in zip(face_locations, face_encodings):
            top, right, bottom, left = location
            face_img = frame[top:bottom, left:right]
            quality = self._assess_quality(face_img)

            if quality > 0.15:  # Seuil de qualité minimum
                self.detected_faces.append(DetectedFace(
                    frame_number=frame_number,
                    location=location,
                    encoding=encoding,
                    quality=quality,
                    image=face_img.copy()
                ))

        return face_locations

    def _assess_quality(self, face_img: np.ndarray) -> float:
        """Évalue la qualité de l'image du visage"""
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.sum()
        contrast_score = np.std(hist_norm) * 100
        return (blur_score * 0.7 + contrast_score * 0.3) / 100

    def assign_face_to_person(self, face_idx: int, person_name: str):
        """Assigne un visage détecté à une personne"""
        if 0 <= face_idx < len(self.detected_faces):
            face = self.detected_faces[face_idx]
            self.selected_faces[person_name].append(face)

    def get_best_faces_for_person(self, person_name: str, max_faces: int = 5) -> List[np.ndarray]:
        """Retourne les meilleurs encodages pour une personne"""
        if person_name not in self.selected_faces:
            return []

        faces = self.selected_faces[person_name]
        # Trier par qualité
        faces.sort(key=lambda x: x.quality, reverse=True)
        return [face.encoding for face in faces[:max_faces]]

    def clear_collections(self):
        """Vide les collections de visages"""
        self.detected_faces.clear()
        self.selected_faces.clear()


def process_video_with_selection(video_path: str, collector: FaceCollector,
                                 progress_callback=None, frame_callback=None):
    """Traite la vidéo et collecte les visages pour sélection"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Impossible d'ouvrir la vidéo")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 10 == 0:  # Traiter une frame sur 10
                face_locations = collector.collect_faces(frame, frame_count)

                if frame_callback:
                    frame_callback(frame, face_locations)

                if progress_callback:
                    progress = (frame_count / total_frames) * 100
                    progress_callback(progress)

    finally:
        cap.release()

    return collector.detected_faces