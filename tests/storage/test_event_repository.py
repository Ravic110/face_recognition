import time

import numpy as np
import pytest

from face_recognition_app.domain.detection import UNKNOWN_NAME, DetectedFace
from face_recognition_app.settings import AppSettings
from face_recognition_app.storage.event_repository import EventRepository


@pytest.fixture
def repo(tmp_path):
    settings = AppSettings.create(project_root=tmp_path)
    settings.ensure_directories()
    r = EventRepository(settings)
    yield r
    r.stop()


def _connu(nom="Alice"):
    return DetectedFace(location=(0, 10, 10, 0), name=nom, confidence=0.9, is_known=True)


def _inconnu():
    return DetectedFace(location=(0, 10, 10, 0), name=UNKNOWN_NAME, confidence=0.1, is_known=False)


def test_enregistrement_et_relecture(repo):
    repo.record_sync(1000.0, "cam1", "Salon", [_connu()])
    events = repo.get_recent()
    assert len(events) == 1
    assert events[0].camera_name == "Salon"
    assert events[0].known_names == ["Alice"]


def test_tri_du_plus_recent_au_plus_ancien(repo):
    repo.record_sync(1000.0, "cam1", "Salon", [])
    repo.record_sync(2000.0, "cam1", "Salon", [])
    assert [e.timestamp for e in repo.get_recent()] == [2000.0, 1000.0]


def test_limite_respectee(repo):
    for i in range(5):
        repo.record_sync(1000.0 + i, "cam1", "Salon", [])
    assert len(repo.get_recent(count=3)) == 3


def test_get_by_id(repo):
    evt = repo.record_sync(1000.0, "cam1", "Salon", [_connu()])
    assert repo.get_by_id(evt.id).camera_name == "Salon"


def test_get_by_id_inexistant(repo):
    assert repo.get_by_id(999) is None


def test_filtre_par_camera(repo):
    repo.record_sync(1000.0, "cam1", "Salon", [])
    repo.record_sync(1001.0, "cam2", "Cuisine", [])
    assert [e.camera_name for e in repo.get_by_camera("cam2")] == ["Cuisine"]


def test_filtre_par_personne(repo):
    repo.record_sync(1000.0, "cam1", "Salon", [_connu("Alice")])
    repo.record_sync(1001.0, "cam1", "Salon", [_connu("Bob")])
    assert [e.known_names for e in repo.get_by_person("Bob")] == [["Bob"]]


def test_filtre_par_personne_echappe_les_jokers_like(repo):
    """Un nom contenant % ne doit pas se comporter comme un joker."""
    repo.record_sync(1000.0, "cam1", "Salon", [_connu("Alice")])
    assert repo.get_by_person("%") == []


def test_comptage(repo):
    repo.record_sync(1000.0, "cam1", "Salon", [])
    repo.record_sync(1001.0, "cam1", "Salon", [])
    assert repo.count() == 2


def test_statistiques_par_camera(repo):
    repo.record_sync(1000.0, "cam1", "Salon", [])
    repo.record_sync(1001.0, "cam2", "Cuisine", [])
    repo.record_sync(1002.0, "cam2", "Cuisine", [])
    stats = repo.stats()
    assert stats["total"] == 3
    assert stats["by_camera"]["Cuisine"] == 2


def test_snapshot_encode_et_redimensionne(repo):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    evt = repo.record_sync(1000.0, "cam1", "Salon", [_inconnu()], frame=frame)
    assert evt.snapshot_b64 is not None
    assert len(evt.snapshot_b64) > 0


def test_pas_de_snapshot_si_desactive(repo):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    evt = repo.record_sync(1000.0, "cam1", "Salon", [], frame=frame, save_snapshot=False)
    assert evt.snapshot_b64 is None


def test_suppression_avant_une_date(repo):
    repo.record_sync(1000.0, "cam1", "Salon", [])
    repo.record_sync(3000.0, "cam1", "Salon", [])
    assert repo.delete_before(2000.0) == 1
    assert repo.count() == 1


def test_purge_selon_la_retention(tmp_path):
    settings = AppSettings.create(project_root=tmp_path)
    settings.ensure_directories()
    repo = EventRepository(settings)
    try:
        ancien = time.time() - (settings.event_retention_days + 1) * 86400
        repo.record_sync(ancien, "cam1", "Salon", [])
        repo.record_sync(time.time(), "cam1", "Salon", [])
        assert repo.purge_expired() == 1
        assert repo.count() == 1
    finally:
        repo.stop()


def test_ecriture_asynchrone_finit_par_persister(repo):
    repo.start()
    repo.record(1000.0, "cam1", "Salon", [_connu()])
    repo.stop()  # draine la file avant de rendre la main
    assert repo.count() == 1


def test_record_sans_thread_ecrit_directement(repo):
    repo.record(1000.0, "cam1", "Salon", [_connu()])
    assert repo.count() == 1


def test_stop_est_idempotent(repo):
    repo.start()
    repo.stop()
    repo.stop()


def test_visages_relus_comme_objets_du_domaine(repo):
    repo.record_sync(1000.0, "cam1", "Salon", [_connu("Alice"), _inconnu()])
    evt = repo.get_recent()[0]
    assert all(isinstance(f, DetectedFace) for f in evt.faces)
    assert evt.unknown_count == 1
    assert evt.has_unknown is True


def test_permissions_restrictives_sur_la_base(tmp_path):
    settings = AppSettings.create(project_root=tmp_path)
    settings.ensure_directories()
    repo = EventRepository(settings)
    try:
        assert settings.events_db.stat().st_mode & 0o077 == 0
    finally:
        repo.stop()
