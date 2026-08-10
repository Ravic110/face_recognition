import logging

import pytest

from face_recognition_app import logging_config
from face_recognition_app.logging_config import configure_logging
from face_recognition_app.settings import AppSettings


@pytest.fixture(autouse=True)
def _reinitialiser_logging():
    """Le logging est global : on restaure l'état racine après chaque test."""
    racine = logging.getLogger()
    handlers, niveau = list(racine.handlers), racine.level
    logging_config._configured = False
    yield
    for h in racine.handlers:
        h.close()
    racine.handlers = handlers
    racine.setLevel(niveau)
    logging_config._configured = False


def test_configure_logging_ecrit_dans_le_fichier(tmp_path):
    settings = AppSettings.create(project_root=tmp_path)
    settings.ensure_directories()
    configure_logging(settings)
    logging.getLogger("test").warning("message de controle")
    for h in logging.getLogger().handlers:
        h.flush()

    contenu = (tmp_path / "app.log").read_text(encoding="utf-8")
    assert "message de controle" in contenu


def test_configure_logging_est_idempotent(tmp_path):
    settings = AppSettings.create(project_root=tmp_path)
    settings.ensure_directories()
    configure_logging(settings)
    avant = len(logging.getLogger().handlers)
    configure_logging(settings)
    assert len(logging.getLogger().handlers) == avant


def test_mode_verbeux_abaisse_le_niveau(tmp_path):
    settings = AppSettings.create(project_root=tmp_path)
    settings.ensure_directories()
    configure_logging(settings, verbose=True)
    assert logging.getLogger().level == logging.DEBUG


def test_werkzeug_reduit_au_silence(tmp_path):
    settings = AppSettings.create(project_root=tmp_path)
    settings.ensure_directories()
    configure_logging(settings)
    assert logging.getLogger("werkzeug").level == logging.ERROR
