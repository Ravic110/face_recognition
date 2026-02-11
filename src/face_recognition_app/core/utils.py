from datetime import datetime
from face_recognition import compare_faces, face_distance

from face_recognition_app.storage.encodings_store import (
    delete_encoding as _delete_encoding,
    load_existing_encodings as _load_existing_encodings,
    save_face_encoding as _save_face_encoding,
    update_metadata_entry as _update_metadata_entry,
    validate_encoding as _validate_encoding,
)


def load_existing_encodings():
    return _load_existing_encodings()


def save_face_encoding(name, encoding, image):
    _save_face_encoding(name, encoding, image)


def is_duplicate(encoding, existing_encodings, tolerance=0.6):
    encodings_list = [e["encoding"] for e in existing_encodings]
    matches = compare_faces(encodings_list, encoding, tolerance + 0.1)

    for i, match in enumerate(matches):
        if match:
            distance = face_distance([existing_encodings[i]["encoding"]], encoding)[0]
            if distance < tolerance:
                return existing_encodings[i]["name"]
    return None


def validate_encoding(data):
    return _validate_encoding(data)


def update_metadata(uid, name):
    _update_metadata_entry(uid, name, datetime.now().isoformat())


def delete_encoding(name):
    return _delete_encoding(name)
