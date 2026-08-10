import json

import pytest

from face_recognition_app.domain.profile import SurveillanceProfile
from face_recognition_app.settings import AppSettings
from face_recognition_app.storage.profile_repository import ProfileRepository


@pytest.fixture
def settings(tmp_path):
    s = AppSettings.create(project_root=tmp_path)
    s.ensure_directories()
    return s


def test_profils_par_defaut_disponibles_sans_fichier(settings):
    repo = ProfileRepository(settings)
    assert {p.name for p in repo.list_all()} == {"present", "absent", "nuit"}


def test_profil_actif_par_defaut(settings):
    assert ProfileRepository(settings).active_name == "present"


def test_changement_de_profil_actif_persiste(settings):
    ProfileRepository(settings).set_active("absent")
    assert ProfileRepository(settings).active_name == "absent"


def test_set_active_sur_profil_inconnu_refuse(settings):
    repo = ProfileRepository(settings)
    assert repo.set_active("inexistant") is False
    assert repo.active_name == "present"


def test_profil_personnalise_persiste(settings):
    repo = ProfileRepository(settings)
    repo.save_profile(SurveillanceProfile(name="vacances", label="Vacances", record_video=True))
    assert ProfileRepository(settings).get("vacances").record_video is True


def test_suppression_d_un_profil_par_defaut_refusee(settings):
    assert ProfileRepository(settings).delete_profile("present") is False


def test_suppression_d_un_profil_personnalise(settings):
    repo = ProfileRepository(settings)
    repo.save_profile(SurveillanceProfile(name="vacances", label="Vacances"))
    assert repo.delete_profile("vacances") is True
    assert repo.get("vacances") is None


def test_suppression_du_profil_actif_bascule_sur_present(settings):
    repo = ProfileRepository(settings)
    repo.save_profile(SurveillanceProfile(name="vacances", label="Vacances"))
    repo.set_active("vacances")
    repo.delete_profile("vacances")
    assert repo.active_name == "present"


def test_fichier_illisible_retombe_sur_les_defauts(settings):
    settings.profiles_file.write_text("{ pas du json")
    repo = ProfileRepository(settings)
    assert repo.active_name == "present"
    assert len(repo.list_all()) == 3


def test_profil_invalide_ignore_les_autres_conserves(settings):
    settings.profiles_file.write_text(
        json.dumps(
            {
                "active": "present",
                "profiles": [
                    {"name": "casse", "label": "Casse", "recognition_threshold": 9.0},
                    {"name": "bon", "label": "Bon"},
                ],
            }
        )
    )
    repo = ProfileRepository(settings)
    assert repo.get("casse") is None
    assert repo.get("bon") is not None


def test_actif_inconnu_retombe_sur_present(settings):
    settings.profiles_file.write_text(json.dumps({"active": "fantome", "profiles": []}))
    assert ProfileRepository(settings).active_name == "present"


def test_profiles_json_existant_relu(settings):
    settings.profiles_file.write_text(
        json.dumps(
            {
                "active": "nuit",
                "profiles": [
                    {
                        "name": "nuit",
                        "label": "Nuit",
                        "record_video": True,
                        "enabled_camera_uids": ["obsolete"],
                    }
                ],
            }
        )
    )
    repo = ProfileRepository(settings)
    assert repo.active_name == "nuit"
    assert repo.get("nuit").record_video is True


def test_migration_importe_les_champs_d_alerte(settings):
    settings.alerts_file.write_text(
        json.dumps(
            {
                "alert_on_unknown": False,
                "alert_on_known": True,
                "target_persons": ["Bob"],
                "smtp_host": "smtp.example.com",
            }
        )
    )
    repo = ProfileRepository(settings)
    assert repo.migrate_alert_fields() is True

    actif = ProfileRepository(settings).get_active()
    assert actif.alert_on_known is True
    assert actif.target_persons == ["Bob"]

    restant = json.loads(settings.alerts_file.read_text())
    assert "alert_on_known" not in restant
    assert restant["smtp_host"] == "smtp.example.com"


def test_migration_sans_fichier_ne_fait_rien(settings):
    assert ProfileRepository(settings).migrate_alert_fields() is False


def test_migration_idempotente(settings):
    settings.alerts_file.write_text(json.dumps({"alert_on_known": True}))
    repo = ProfileRepository(settings)
    assert repo.migrate_alert_fields() is True
    assert repo.migrate_alert_fields() is False
