import numpy as np
import pytest

from face_recognition_app.storage import encodings_store as store


def _setup_tmp_store(tmp_path):
    store.ENCODED_DIR = str(tmp_path)
    store.META_FILE = str(tmp_path / "metadata.json")


# ---------------------------------------------------------------------------
# encodings_store
# ---------------------------------------------------------------------------

def test_save_load_delete_encoding(tmp_path):
    _setup_tmp_store(tmp_path)

    encoding = np.zeros(128, dtype=float)
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    store.save_face_encoding("Alice", encoding, image=image)

    encodings = store.load_existing_encodings()
    assert any(e["name"] == "Alice" for e in encodings)

    loaded_image = store.load_image_for_name("Alice")
    assert loaded_image is not None

    removed = store.delete_encoding("Alice")
    assert removed

    encodings_after = store.load_existing_encodings()
    assert not any(e["name"] == "Alice" for e in encodings_after)


def test_save_without_image(tmp_path):
    _setup_tmp_store(tmp_path)

    encoding = np.ones(128, dtype=float)
    store.save_face_encoding("Bob", encoding)

    encodings = store.load_existing_encodings()
    assert any(e["name"] == "Bob" for e in encodings)

    # Aucune image stockée → load_image_for_name doit retourner None
    img = store.load_image_for_name("Bob")
    assert img is None


def test_load_encodings_map(tmp_path):
    _setup_tmp_store(tmp_path)

    enc1 = np.zeros(128, dtype=float)
    enc2 = np.ones(128, dtype=float)
    store.save_face_encoding("Alice", enc1)
    store.save_face_encoding("Bob", enc2)

    mapping = store.load_encodings_map()
    assert "Alice" in mapping
    assert "Bob" in mapping
    assert mapping["Alice"].shape == (128,)


def test_load_encodings_map_deduplication(tmp_path):
    """Si deux fichiers portent le même nom, load_encodings_map ne garde que le premier."""
    _setup_tmp_store(tmp_path)

    enc = np.zeros(128, dtype=float)
    store.save_face_encoding("Alice", enc)
    store.save_face_encoding("Alice", enc)

    mapping = store.load_encodings_map()
    assert list(mapping.keys()).count("Alice") == 1


def test_delete_nonexistent_encoding(tmp_path):
    _setup_tmp_store(tmp_path)
    removed = store.delete_encoding("Inconnu")
    assert removed == []


def test_validate_encoding_valid():
    data = {"name": "X", "encoding": [0.0] * 128, "timestamp": "2024-01-01T00:00:00"}
    assert store.validate_encoding(data) is True


def test_validate_encoding_missing_key():
    data = {"name": "X", "encoding": [0.0] * 128}  # pas de timestamp
    assert store.validate_encoding(data) is False


def test_validate_encoding_wrong_length():
    data = {"name": "X", "encoding": [0.0] * 64, "timestamp": "2024-01-01T00:00:00"}
    assert store.validate_encoding(data) is False


def test_metadata_updated_on_save(tmp_path):
    _setup_tmp_store(tmp_path)

    encoding = np.zeros(128, dtype=float)
    store.save_face_encoding("Charlie", encoding)

    metadata = store.load_metadata()
    names = [v["name"] for v in metadata.values()]
    assert "Charlie" in names


def test_metadata_cleaned_on_delete(tmp_path):
    _setup_tmp_store(tmp_path)

    encoding = np.zeros(128, dtype=float)
    store.save_face_encoding("Dave", encoding)
    store.delete_encoding("Dave")

    metadata = store.load_metadata()
    names = [v["name"] for v in metadata.values()]
    assert "Dave" not in names


# ---------------------------------------------------------------------------
# utils (is_duplicate)
# ---------------------------------------------------------------------------

from face_recognition_app.core.utils import is_duplicate
from face_recognition_app.storage.config import DUPLICATE_TOLERANCE


def _make_existing(name, vec):
    return [{"name": name, "encoding": np.array(vec, dtype=float)}]


def test_is_duplicate_identical():
    vec = [0.1] * 128
    existing = _make_existing("Alice", vec)
    result = is_duplicate(np.array(vec, dtype=float), existing)
    assert result == "Alice"


def test_is_duplicate_no_match():
    vec_a = np.zeros(128, dtype=float)
    vec_b = np.ones(128, dtype=float)  # très différent
    existing = _make_existing("Alice", vec_b)
    result = is_duplicate(vec_a, existing)
    assert result is None


def test_is_duplicate_empty_existing():
    vec = np.zeros(128, dtype=float)
    result = is_duplicate(vec, [])
    assert result is None


def test_is_duplicate_respects_custom_tolerance():
    """Avec une tolerance très faible, même un vecteur presque identique ne doit pas matcher."""
    base = np.zeros(128, dtype=float)
    slightly_different = base.copy()
    slightly_different[0] = 0.3
    existing = _make_existing("Alice", slightly_different)
    result = is_duplicate(base, existing, tolerance=0.01)
    assert result is None
