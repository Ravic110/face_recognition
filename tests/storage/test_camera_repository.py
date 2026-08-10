import json

import pytest

from face_recognition_app.domain.camera import CameraConfig
from face_recognition_app.settings import AppSettings
from face_recognition_app.storage.camera_repository import CameraRepository


@pytest.fixture
def repo(tmp_path):
    settings = AppSettings.create(project_root=tmp_path)
    settings.ensure_directories()
    return CameraRepository(settings)


def test_liste_vide_sans_fichier(repo):
    assert repo.load_all() == []


def test_aller_retour(repo):
    configs = [
        CameraConfig(name="Salon", source_type="webcam", source=0),
        CameraConfig(name="Couloir", source_type="ip", source="http://10.0.0.5:8080/video"),
    ]
    repo.save_all(configs)
    assert [c.name for c in repo.load_all()] == ["Salon", "Couloir"]


def test_fichier_illisible_retourne_liste_vide(repo, tmp_path):
    (tmp_path / "cameras.json").write_text("{ pas du json")
    assert repo.load_all() == []


def test_entree_invalide_ignoree_les_autres_conservees(repo, tmp_path):
    (tmp_path / "cameras.json").write_text(
        json.dumps(
            [
                {"name": "Bonne", "source_type": "webcam", "source": 0},
                {"name": "Mauvaise", "source_type": "satellite", "source": 0},
            ]
        )
    )
    assert [c.name for c in repo.load_all()] == ["Bonne"]


def test_entree_sans_champ_obligatoire_ignoree(repo, tmp_path):
    (tmp_path / "cameras.json").write_text(
        json.dumps(
            [
                {"source_type": "webcam", "source": 0},
                {"name": "Bonne", "source_type": "webcam", "source": 0},
            ]
        )
    )
    assert [c.name for c in repo.load_all()] == ["Bonne"]


def test_format_existant_relu(repo, tmp_path):
    (tmp_path / "cameras.json").write_text(
        json.dumps(
            [
                {
                    "uid": "6dc1bce5",
                    "name": "camera couloir",
                    "source_type": "ip",
                    "source": "http://10.18.154.148:8080/video",
                    "enabled": True,
                    "width": 640,
                    "height": 480,
                    "roi": None,
                    "detection_model": "hog",
                }
            ]
        )
    )
    configs = repo.load_all()
    assert len(configs) == 1
    assert configs[0].uid == "6dc1bce5"
