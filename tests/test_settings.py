import dataclasses

import pytest

from face_recognition_app.settings import AppSettings


def test_chemins_derives_de_la_racine(tmp_path):
    s = AppSettings.create(project_root=tmp_path)
    assert s.encodings_dir == tmp_path / "encodings"
    assert s.events_db == tmp_path / "events.db"
    assert s.clips_dir == tmp_path / "clips"
    assert s.cameras_file == tmp_path / "cameras.json"
    assert s.profiles_file == tmp_path / "profiles.json"


def test_valeurs_par_defaut_sures(tmp_path):
    s = AppSettings.create(project_root=tmp_path)
    assert s.api_host == "127.0.0.1"
    assert s.api_allow_control is False
    assert s.event_retention_days == 30
    assert s.capture_fps == 15.0


def test_surcharge_par_variables_d_environnement(tmp_path, monkeypatch):
    monkeypatch.setenv("FR_API_PORT", "8123")
    monkeypatch.setenv("FR_API_HOST", "0.0.0.0")
    monkeypatch.setenv("FR_EVENT_RETENTION_DAYS", "7")
    s = AppSettings.create(project_root=tmp_path)
    assert s.api_port == 8123
    assert s.api_host == "0.0.0.0"
    assert s.event_retention_days == 7


def test_variable_invalide_conserve_le_defaut(tmp_path, monkeypatch):
    monkeypatch.setenv("FR_API_PORT", "pas-un-nombre")
    s = AppSettings.create(project_root=tmp_path)
    assert s.api_port == 5000


def test_ensure_directories_cree_les_dossiers(tmp_path):
    s = AppSettings.create(project_root=tmp_path)
    s.ensure_directories()
    assert s.encodings_dir.is_dir()
    assert s.clips_dir.is_dir()
    assert s.config_dir.is_dir()


def test_settings_est_immuable(tmp_path):
    s = AppSettings.create(project_root=tmp_path)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.api_port = 1234
