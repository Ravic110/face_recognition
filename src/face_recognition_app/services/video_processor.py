import cv2
import face_recognition
import os
import json
import uuid

from face_recognition_app.storage.config import ENCODED_DIR


def process_chunk(video_path, start_frame, end_frame, chunk_id, frame_skip):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    temp_dir = os.path.join(ENCODED_DIR, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    output_file = os.path.join(temp_dir, f'chunk_{chunk_id}.json')
    results = []

    frame_count = start_frame
    while frame_count <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Ne traiter que si frame_count est un multiple de frame_skip
        if frame_count % frame_skip == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                try:
                    face_image = frame[top:bottom, left:right]
                    _, buffer = cv2.imencode('.jpg', face_image)

                    results.append({
                        'id': f"temp_{uuid.uuid4().hex}",
                        'encoding': encoding.tolist(),
                        'image': buffer.tobytes().hex()  # Sérialisation hexadécimale
                    })
                except Exception:
                    continue

        frame_count += 1

    with open(output_file, 'w') as f:
        json.dump(results, f)

    cap.release()
    return output_file
