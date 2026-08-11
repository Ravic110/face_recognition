import threading
import time

import numpy as np
import pytest

from face_recognition_app.domain.camera import CameraConfig
from face_recognition_app.domain.connection import ConnectionState, ReconnectPolicy
from face_recognition_app.services.camera_source import CameraSource

# Backoff minuscule : les tests ne doivent jamais attendre réellement.
POLITIQUE_RAPIDE = ReconnectPolicy(delay_min=0.01, delay_max=0.02, factor=2.0)


def _config():
    return CameraConfig(name="Test", source_type="webcam", source=0)


class SourceSimulee(CameraSource):
    """
    Source contrôlée par le test — aucune caméra réelle.

    `ouvertures_echouees` : nombre d'ouvertures qui échouent avant de réussir.
    `lectures_avant_perte` : nombre de lectures réussies avant une perte de flux.
    """

    def __init__(self, config, policy=None, ouvertures_echouees=0, lectures_avant_perte=None):
        super().__init__(config, policy=policy)
        self.ouvertures_echouees = ouvertures_echouees
        self.lectures_avant_perte = lectures_avant_perte
        self.ouvertures_tentees = 0
        self.liberations = 0
        self._lectures = 0
        self._ouvert = False

    def _open_capture(self) -> bool:
        self.ouvertures_tentees += 1
        if self.ouvertures_tentees <= self.ouvertures_echouees:
            return False
        self._ouvert = True
        self._lectures = 0
        return True

    def _read_raw(self):
        if not self._ouvert:
            return False, None
        if self.lectures_avant_perte is not None and self._lectures >= self.lectures_avant_perte:
            return False, None
        self._lectures += 1
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def _release(self) -> None:
        self.liberations += 1
        self._ouvert = False


def _attendre(predicat, timeout=2.0):
    """Attend qu'une condition devienne vraie, sans dormir inutilement."""
    limite = time.monotonic() + timeout
    while time.monotonic() < limite:
        if predicat():
            return True
        time.sleep(0.005)
    return False


def test_demarre_et_fournit_une_frame():
    src = SourceSimulee(_config(), policy=POLITIQUE_RAPIDE)
    assert src.start() is True
    try:
        assert _attendre(lambda: src.get_frame() is not None)
        assert src.state is ConnectionState.CONNECTED
        assert src.is_connected is True
    finally:
        src.stop()


def test_echec_d_ouverture_initiale():
    src = SourceSimulee(_config(), policy=POLITIQUE_RAPIDE, ouvertures_echouees=99)
    assert src.start() is False
    assert src.state is ConnectionState.DISCONNECTED
    assert src.is_running is False


def test_reconnexion_apres_perte_de_flux():
    src = SourceSimulee(_config(), policy=POLITIQUE_RAPIDE, lectures_avant_perte=2)
    src.start()
    try:
        assert _attendre(lambda: src.ouvertures_tentees >= 3)
        assert src.state is ConnectionState.CONNECTED
    finally:
        src.stop()


def test_reconnexion_persiste_apres_plusieurs_echecs():
    """
    Le defaut historique : apres un echec de reouverture, la boucle restait
    inerte et la camera ne revenait jamais.
    """
    src = SourceSimulee(_config(), policy=POLITIQUE_RAPIDE, lectures_avant_perte=1)
    src.start()
    try:
        assert _attendre(lambda: src.ouvertures_tentees >= 5, timeout=3.0)
    finally:
        src.stop()


def test_ouverture_ratee_libere_la_capture():
    """Une capture ouverte mais inutilisable ne doit pas fuir."""
    src = SourceSimulee(_config(), policy=POLITIQUE_RAPIDE, lectures_avant_perte=0)
    src.start()
    try:
        assert _attendre(lambda: src.liberations >= 1)
    finally:
        src.stop()


def test_stop_rend_la_main_pendant_l_attente_de_reconnexion():
    """Le sleep etait bloquant jusqu'a 60s ; stop() doit interrompre l'attente."""
    lente = ReconnectPolicy(delay_min=30.0, delay_max=60.0)
    src = SourceSimulee(_config(), policy=lente, lectures_avant_perte=1, ouvertures_echouees=0)
    src.start()
    try:
        assert _attendre(lambda: src.state is not ConnectionState.CONNECTED, timeout=2.0)
    finally:
        debut = time.monotonic()
        src.stop()
        assert time.monotonic() - debut < 2.0


def test_stop_est_idempotent():
    src = SourceSimulee(_config(), policy=POLITIQUE_RAPIDE)
    src.start()
    src.stop()
    src.stop()
    assert src.is_running is False


def test_start_est_idempotent():
    src = SourceSimulee(_config(), policy=POLITIQUE_RAPIDE)
    try:
        assert src.start() is True
        assert src.start() is True
        assert _attendre(lambda: src.get_frame() is not None)
        assert threading.active_count() < 20
    finally:
        src.stop()


def test_le_compteur_de_tentatives_se_reinitialise_apres_succes():
    src = SourceSimulee(_config(), policy=POLITIQUE_RAPIDE, lectures_avant_perte=3)
    src.start()
    try:
        assert _attendre(lambda: src.ouvertures_tentees >= 2)
        assert _attendre(lambda: src.reconnect_attempts == 0)
    finally:
        src.stop()


def test_next_retry_in_est_nul_quand_connecte():
    src = SourceSimulee(_config(), policy=POLITIQUE_RAPIDE)
    src.start()
    try:
        assert _attendre(lambda: src.state is ConnectionState.CONNECTED)
        assert src.next_retry_in == pytest.approx(0.0)
    finally:
        src.stop()


def test_get_frame_retourne_une_copie():
    src = SourceSimulee(_config(), policy=POLITIQUE_RAPIDE)
    src.start()
    try:
        assert _attendre(lambda: src.get_frame() is not None)
        a, b = src.get_frame(), src.get_frame()
        assert a is not b
    finally:
        src.stop()


# ── Effet miroir ──────────────────────────────────────────────────────────────


class CaptureFactice:
    """Objet minimal imitant cv2.VideoCapture pour tester `_read_raw`."""

    def __init__(self, frame):
        self._frame = frame

    def read(self):
        return True, self._frame

    def release(self):
        pass


class SourceAvecCapture(CameraSource):
    """Source qui expose une capture factice, sans toucher a OpenCV."""

    def __init__(self, config, frame):
        super().__init__(config, policy=POLITIQUE_RAPIDE)
        self._frame_test = frame

    def _open_capture(self) -> bool:
        self._cap = CaptureFactice(self._frame_test)
        return True


def _frame_asymetrique():
    """Frame dont la colonne 0 differe de la derniere : le miroir se voit."""
    f = np.zeros((2, 4, 3), dtype=np.uint8)
    f[:, 0] = 255
    return f


def test_webcam_retourne_horizontalement_la_frame():
    src = SourceAvecCapture(
        CameraConfig(name="W", source_type="webcam", source=0), _frame_asymetrique()
    )
    src._open_capture()
    ok, frame = src._read_raw()
    assert ok
    # La colonne blanche est passee de gauche a droite
    assert frame[0, 0].tolist() == [0, 0, 0]
    assert frame[0, -1].tolist() == [255, 255, 255]


def test_camera_ip_ne_retourne_pas_la_frame():
    src = SourceAvecCapture(
        CameraConfig(name="I", source_type="ip", source="http://x"), _frame_asymetrique()
    )
    src._open_capture()
    ok, frame = src._read_raw()
    assert ok
    assert frame[0, 0].tolist() == [255, 255, 255]


def test_miroir_desactivable_sur_une_webcam():
    src = SourceAvecCapture(
        CameraConfig(name="W", source_type="webcam", source=0, mirror=False),
        _frame_asymetrique(),
    )
    src._open_capture()
    _, frame = src._read_raw()
    assert frame[0, 0].tolist() == [255, 255, 255]


def test_lecture_ratee_ne_tente_pas_de_retourner():
    class CaptureMuette(CaptureFactice):
        def read(self):
            return False, None

    src = SourceAvecCapture(CameraConfig(name="W", source_type="webcam", source=0), None)
    src._cap = CaptureMuette(None)
    ok, frame = src._read_raw()
    assert ok is False
    assert frame is None
