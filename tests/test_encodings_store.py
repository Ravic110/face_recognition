import numpy as np

import encodings_store as store


def _setup_tmp_store(tmp_path):
    store.ENCODED_DIR = str(tmp_path)
    store.META_FILE = str(tmp_path / "metadata.json")


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
