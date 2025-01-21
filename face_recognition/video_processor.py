import os
import cv2
import face_recognition
from utils import save_face_encoding, load_existing_encodings, is_duplicate

def process_video(video_path, tolerance=0.5, frame_skip=30):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"La vidéo {video_path} n'existe pas.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Impossible d'ouvrir la vidéo.")

    print(f"Traitement de la vidéo : {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frames += 1
        if processed_frames % frame_skip != 0:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        faces = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            face_image = frame[top:bottom, left:right]
            faces.append((face_image, face_encoding))

        yield faces  # Retourne tous les visages de la frame comme une liste

    cap.release()
    print("Traitement vidéo terminé.")

