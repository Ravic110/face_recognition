import base64
import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .config import ENCODED_DIR, META_FILE


def _iter_encoding_files():
    meta_name = Path(META_FILE).name
    for filename in os.listdir(ENCODED_DIR):
        if not filename.endswith(".json"):
            continue
        if filename == meta_name:
            continue
        if "temp" in filename:
            continue
        yield os.path.join(ENCODED_DIR, filename)


def _read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def load_metadata():
    if os.path.exists(META_FILE):
        try:
            return _read_json(META_FILE)
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def update_metadata_entry(uid, name, timestamp):
    metadata = load_metadata()
    metadata[uid] = {
        "name": name,
        "date_creation": timestamp,
    }
    _write_json(META_FILE, metadata)


def load_existing_encodings():
    encodings = []
    for filepath in _iter_encoding_files():
        try:
            data = _read_json(filepath)
        except (OSError, json.JSONDecodeError):
            continue

        if not validate_encoding(data):
            continue

        encodings.append({
            "name": data["name"],
            "encoding": np.array(data["encoding"]),
        })
    return encodings


def load_encodings_map():
    encodings = {}
    for entry in load_existing_encodings():
        if entry["name"] not in encodings:
            encodings[entry["name"]] = entry["encoding"]
    return encodings


def save_face_encoding(name, encoding, image=None):
    timestamp = datetime.now().isoformat()
    unique_id = uuid.uuid4().hex[:12]

    data = {
        "name": name,
        "encoding": encoding.tolist(),
        "timestamp": timestamp,
    }

    if image is not None:
        _, buffer = cv2.imencode(".jpg", image)
        image_base64 = base64.b64encode(buffer).decode("utf-8")
        data["image_base64"] = image_base64

    os.makedirs(ENCODED_DIR, exist_ok=True)
    _write_json(os.path.join(ENCODED_DIR, f"{unique_id}.json"), data)
    update_metadata_entry(unique_id, name, timestamp)


def validate_encoding(data):
    required = ["name", "encoding", "timestamp"]
    if not all(key in data for key in required):
        return False
    return isinstance(data["encoding"], list) and len(data["encoding"]) == 128


def _decode_image_base64(image_base64):
    if not image_base64:
        return None
    try:
        image_bytes = base64.b64decode(image_base64)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    except Exception:
        return None


def load_image_for_name(name):
    for filepath in _iter_encoding_files():
        try:
            data = _read_json(filepath)
        except (OSError, json.JSONDecodeError):
            continue

        if data.get("name") != name:
            continue

        if "image_base64" in data:
            return _decode_image_base64(data.get("image_base64"))

        if "image" in data and isinstance(data["image"], str):
            return _decode_image_base64(data.get("image"))

    return None


def delete_encoding(name):
    removed = []
    for filepath in _iter_encoding_files():
        try:
            data = _read_json(filepath)
        except (OSError, json.JSONDecodeError):
            continue

        if data.get("name") == name:
            try:
                os.remove(filepath)
                removed.append(Path(filepath).stem)
            except OSError:
                continue

    if removed:
        metadata = load_metadata()
        for uid in removed:
            metadata.pop(uid, None)
        _write_json(META_FILE, metadata)

    return removed
