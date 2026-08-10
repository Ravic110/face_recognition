import time

import pytest

from face_recognition_app.api.auth import ApiKeyGuard, resolve_api_key
from face_recognition_app.settings import AppSettings


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.delenv("FR_API_KEY", raising=False)
    s = AppSettings.create(project_root=tmp_path)
    s.ensure_directories()
    return s


# ── Résolution de la clé ──────────────────────────────────────────────────────


def test_cle_generee_au_premier_appel(settings):
    cle = resolve_api_key(settings)
    assert len(cle) >= 32
    assert settings.api_key_file.exists()


def test_cle_stable_entre_deux_appels(settings):
    assert resolve_api_key(settings) == resolve_api_key(settings)


def test_cle_fichier_en_permissions_restrictives(settings):
    resolve_api_key(settings)
    assert settings.api_key_file.stat().st_mode & 0o077 == 0


def test_variable_d_environnement_prioritaire(settings, monkeypatch):
    monkeypatch.setenv("FR_API_KEY", "cle-de-l-environnement")
    assert resolve_api_key(settings) == "cle-de-l-environnement"


def test_variable_d_environnement_n_ecrase_pas_le_fichier(settings, monkeypatch):
    depuis_fichier = resolve_api_key(settings)
    monkeypatch.setenv("FR_API_KEY", "temporaire")
    assert resolve_api_key(settings) == "temporaire"
    monkeypatch.delenv("FR_API_KEY")
    assert resolve_api_key(settings) == depuis_fichier


def test_fichier_vide_regenere_une_cle(settings):
    settings.api_key_file.write_text("   ")
    assert len(resolve_api_key(settings)) >= 32


# ── Vérification ──────────────────────────────────────────────────────────────


def test_bonne_cle_acceptee():
    assert ApiKeyGuard("secret").check("secret", "10.0.0.1") is True


def test_mauvaise_cle_refusee():
    assert ApiKeyGuard("secret").check("autre", "10.0.0.1") is False


def test_cle_absente_refusee():
    assert ApiKeyGuard("secret").check(None, "10.0.0.1") is False


def test_cle_vide_refusee():
    assert ApiKeyGuard("secret").check("", "10.0.0.1") is False


# ── Limitation des tentatives ─────────────────────────────────────────────────


def test_blocage_apres_le_quota_d_echecs():
    guard = ApiKeyGuard("secret", max_attempts=3)
    for _ in range(3):
        guard.check("faux", "10.0.0.1")
    assert guard.is_blocked("10.0.0.1") is True


def test_ip_bloquee_refusee_meme_avec_la_bonne_cle():
    guard = ApiKeyGuard("secret", max_attempts=2)
    guard.check("faux", "10.0.0.1")
    guard.check("faux", "10.0.0.1")
    assert guard.check("secret", "10.0.0.1") is False


def test_le_blocage_est_par_adresse():
    guard = ApiKeyGuard("secret", max_attempts=2)
    guard.check("faux", "10.0.0.1")
    guard.check("faux", "10.0.0.1")
    assert guard.is_blocked("10.0.0.2") is False
    assert guard.check("secret", "10.0.0.2") is True


def test_un_succes_remet_le_compteur_a_zero():
    guard = ApiKeyGuard("secret", max_attempts=3)
    guard.check("faux", "10.0.0.1")
    guard.check("secret", "10.0.0.1")
    guard.check("faux", "10.0.0.1")
    assert guard.is_blocked("10.0.0.1") is False


def test_le_blocage_expire_avec_la_fenetre():
    guard = ApiKeyGuard("secret", max_attempts=2, window_seconds=0.05)
    guard.check("faux", "10.0.0.1")
    guard.check("faux", "10.0.0.1")
    assert guard.is_blocked("10.0.0.1") is True

    time.sleep(0.06)
    assert guard.is_blocked("10.0.0.1") is False


def test_reset_debloque():
    guard = ApiKeyGuard("secret", max_attempts=1)
    guard.check("faux", "10.0.0.1")
    guard.reset("10.0.0.1")
    assert guard.is_blocked("10.0.0.1") is False
