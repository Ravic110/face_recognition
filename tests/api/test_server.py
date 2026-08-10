import numpy as np
import pytest

from face_recognition_app.api.server import ApiServer
from face_recognition_app.domain.camera import CameraConfig
from face_recognition_app.domain.detection import DetectedFace
from face_recognition_app.settings import AppSettings
from face_recognition_app.storage.event_repository import EventRepository


class FauxCameraManager:
    def __init__(self):
        self._configs = [CameraConfig(name="Salon", source_type="webcam", source=0, uid="cam1")]
        self.demarrages = 0
        self.arrets = 0

    def list_configs(self):
        return self._configs

    def get_config(self, uid):
        return next((c for c in self._configs if c.uid == uid), None)

    def is_running(self, uid):
        return uid == "cam1"

    def get_all_sources(self):
        return {"cam1": object()}

    def get_frame(self, uid):
        return np.zeros((48, 64, 3), dtype=np.uint8) if uid == "cam1" else None

    def start_all(self):
        self.demarrages += 1

    def stop_all(self):
        self.arrets += 1


class FauxMoteur:
    def __init__(self):
        self.actif = False

    @property
    def is_running(self):
        return self.actif

    def start(self):
        self.actif = True

    def stop(self):
        self.actif = False


class FauxRecorder:
    def list_clips(self):
        return []


def _serveur(tmp_path):
    settings = AppSettings.create(project_root=tmp_path)
    settings.ensure_directories()
    events = EventRepository(settings)
    return (
        ApiServer(settings, FauxCameraManager(), FauxMoteur(), events, FauxRecorder()),
        events,
        settings,
    )


@pytest.fixture
def contexte(tmp_path, monkeypatch):
    monkeypatch.setenv("FR_API_KEY", "cle-de-test")
    serveur, events, settings = _serveur(tmp_path)
    yield serveur, events, settings
    serveur.stop()
    events.stop()


@pytest.fixture
def client(contexte):
    serveur, _, _ = contexte
    return serveur.app.test_client()


EN_TETES = {"X-API-Key": "cle-de-test"}


# ── Authentification ──────────────────────────────────────────────────────────


def test_sans_cle_refuse(client):
    assert client.get("/api/status").status_code == 401


def test_mauvaise_cle_refuse(client):
    assert client.get("/api/status", headers={"X-API-Key": "faux"}).status_code == 401


def test_bonne_cle_acceptee(client):
    assert client.get("/api/status", headers=EN_TETES).status_code == 200


def test_toutes_les_routes_de_lecture_sont_protegees(client):
    for route in ("/api/status", "/api/cameras", "/api/events", "/api/faces", "/api/clips"):
        assert client.get(route).status_code == 401, route


def test_blocage_apres_trop_d_echecs(client):
    for _ in range(11):
        client.get("/api/status", headers={"X-API-Key": "faux"})
    assert client.get("/api/status", headers=EN_TETES).status_code == 429


# ── Lecture ───────────────────────────────────────────────────────────────────


def test_status(client):
    data = client.get("/api/status", headers=EN_TETES).get_json()
    assert data["cameras_total"] == 1
    assert data["surveillance_active"] is False


def test_liste_des_cameras(client):
    data = client.get("/api/cameras", headers=EN_TETES).get_json()
    assert data[0]["name"] == "Salon"
    assert data[0]["running"] is True


def test_snapshot_retourne_du_jpeg(client):
    r = client.get("/api/snapshot/cam1", headers=EN_TETES)
    assert r.status_code == 200
    assert r.mimetype == "image/jpeg"


def test_snapshot_camera_inconnue(client):
    assert client.get("/api/snapshot/absente", headers=EN_TETES).status_code == 404


def test_evenements(contexte, client):
    _, events, _ = contexte
    events.record_sync(
        1000.0,
        "cam1",
        "Salon",
        [DetectedFace(location=(0, 1, 1, 0), name="Alice", confidence=0.9, is_known=True)],
    )
    data = client.get("/api/events", headers=EN_TETES).get_json()
    assert len(data) == 1
    assert data[0]["faces"][0]["name"] == "Alice"


def test_detail_d_un_evenement(contexte, client):
    _, events, _ = contexte
    evt = events.record_sync(1000.0, "cam1", "Salon", [])
    data = client.get(f"/api/events/{evt.id}", headers=EN_TETES).get_json()
    assert data["camera_name"] == "Salon"


def test_evenement_inconnu(client):
    assert client.get("/api/events/9999", headers=EN_TETES).status_code == 404


def test_limite_des_evenements_plafonnee(client):
    assert client.get("/api/events?limit=99999", headers=EN_TETES).status_code == 200


def test_limite_non_numerique_rejetee(client):
    assert client.get("/api/events?limit=beaucoup", headers=EN_TETES).status_code == 400


# ── Écriture, désactivée par défaut ───────────────────────────────────────────


def test_controle_refuse_par_defaut(client):
    assert client.post("/api/surveillance/start", headers=EN_TETES).status_code == 403


def test_controle_autorise_quand_active(tmp_path, monkeypatch):
    monkeypatch.setenv("FR_API_KEY", "cle-de-test")
    monkeypatch.setenv("FR_API_ALLOW_CONTROL", "true")
    serveur, events, _ = _serveur(tmp_path)
    try:
        client = serveur.app.test_client()
        assert client.post("/api/surveillance/start", headers=EN_TETES).status_code == 200
        assert serveur._engine.is_running is True
        assert client.post("/api/surveillance/stop", headers=EN_TETES).status_code == 200
        assert serveur._engine.is_running is False
    finally:
        serveur.stop()
        events.stop()


def test_controle_exige_aussi_la_cle(tmp_path, monkeypatch):
    monkeypatch.setenv("FR_API_KEY", "cle-de-test")
    monkeypatch.setenv("FR_API_ALLOW_CONTROL", "true")
    serveur, events, _ = _serveur(tmp_path)
    try:
        assert serveur.app.test_client().post("/api/surveillance/start").status_code == 401
    finally:
        serveur.stop()
        events.stop()


# ── Cycle de vie ──────────────────────────────────────────────────────────────


@pytest.fixture
def serveur_ephemere(tmp_path, monkeypatch):
    """Serveur lie sur un port choisi par l'OS, pour ne pas dependre du 5000."""
    monkeypatch.setenv("FR_API_KEY", "cle-de-test")
    monkeypatch.setenv("FR_API_PORT", "0")
    serveur, events, _ = _serveur(tmp_path)
    yield serveur
    serveur.stop()
    events.stop()


def test_ecoute_locale_par_defaut(contexte):
    serveur, _, settings = contexte
    assert settings.api_host == "127.0.0.1"
    assert serveur.url.startswith("http://127.0.0.1:")


def test_demarrage_puis_arret_reel(serveur_ephemere):
    assert serveur_ephemere.start() is True
    assert serveur_ephemere.is_running is True
    serveur_ephemere.stop()
    assert serveur_ephemere.is_running is False


def test_port_reellement_lie_expose(serveur_ephemere):
    serveur_ephemere.start()
    assert serveur_ephemere.port > 0
    assert f":{serveur_ephemere.port}" in serveur_ephemere.url


def test_redemarrage_possible_apres_arret(serveur_ephemere):
    """Le defaut historique : stop() ne fermait rien et is_running restait vrai."""
    serveur_ephemere.start()
    serveur_ephemere.stop()
    assert serveur_ephemere.start() is True
    assert serveur_ephemere.is_running is True


def test_start_et_stop_idempotents(serveur_ephemere):
    serveur_ephemere.start()
    serveur_ephemere.start()
    serveur_ephemere.stop()
    serveur_ephemere.stop()
    assert serveur_ephemere.is_running is False
