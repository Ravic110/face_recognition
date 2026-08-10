import json

import numpy as np
import pytest

from face_recognition_app.settings import AppSettings
from face_recognition_app.storage.encodings_repository import EncodingsRepository


@pytest.fixture
def repo(tmp_path):
    settings = AppSettings.create(project_root=tmp_path)
    settings.ensure_directories()
    return EncodingsRepository(settings)


def test_sauvegarde_puis_relecture(repo):
    uid = repo.save("Alice", np.zeros(128))
    assert len(uid) == 12
    assert [e.name for e in repo.load_all()] == ["Alice"]


def test_sauvegarde_avec_image_relisible(repo):
    repo.save("Alice", np.zeros(128), image=np.zeros((50, 50, 3), dtype=np.uint8))
    assert repo.load_image("Alice") is not None


def test_sans_image_load_image_retourne_none(repo):
    repo.save("Bob", np.ones(128))
    assert repo.load_image("Bob") is None


def test_suppression(repo):
    repo.save("Alice", np.zeros(128))
    assert repo.delete("Alice")
    assert repo.load_all() == []


def test_suppression_inexistant_retourne_liste_vide(repo):
    assert repo.delete("Personne") == []


def test_load_map_dedoublonne_les_noms(repo):
    repo.save("Alice", np.zeros(128))
    repo.save("Alice", np.zeros(128))
    assert list(repo.load_map()) == ["Alice"]


def test_load_map_retourne_des_vecteurs_128(repo):
    repo.save("Alice", np.zeros(128))
    assert repo.load_map()["Alice"].shape == (128,)


def test_metadata_mise_a_jour_a_la_sauvegarde(repo):
    repo.save("Charlie", np.zeros(128))
    assert "Charlie" in [v["name"] for v in repo.load_metadata().values()]


def test_metadata_nettoyee_a_la_suppression(repo):
    repo.save("Dave", np.zeros(128))
    repo.delete("Dave")
    assert "Dave" not in [v["name"] for v in repo.load_metadata().values()]


def test_fichier_corrompu_ignore_sans_planter(repo, tmp_path):
    repo.save("Alice", np.zeros(128))
    (tmp_path / "encodings" / "corrompu.json").write_text("{ pas du json")
    assert [e.name for e in repo.load_all()] == ["Alice"]


def test_encodage_de_mauvaise_longueur_ignore(repo, tmp_path):
    repo.save("Alice", np.zeros(128))
    (tmp_path / "encodings" / "court.json").write_text(
        json.dumps({"name": "Court", "encoding": [0.0] * 64, "timestamp": "2026-01-01T00:00:00"})
    )
    assert [e.name for e in repo.load_all()] == ["Alice"]


def test_save_rejette_un_encodage_de_mauvaise_taille(repo):
    with pytest.raises(ValueError, match="128"):
        repo.save("Alice", np.zeros(64))


def test_find_duplicate_name(repo):
    vec = np.full(128, 0.1)
    repo.save("Alice", vec)
    assert repo.find_duplicate_name(vec) == "Alice"
    assert repo.find_duplicate_name(np.ones(128)) is None


def test_permissions_restrictives_sur_les_encodages(repo, tmp_path):
    """Donnees biometriques : lisibles par le seul proprietaire."""
    repo.save("Alice", np.zeros(128))
    fichier = next(p for p in (tmp_path / "encodings").glob("*.json") if p.name != "metadata.json")
    assert fichier.stat().st_mode & 0o077 == 0
