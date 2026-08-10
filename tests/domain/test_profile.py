import pytest

from face_recognition_app.domain.detection import UNKNOWN_NAME, DetectedFace
from face_recognition_app.domain.profile import DEFAULT_PROFILES, SurveillanceProfile


def _connu(nom="Alice"):
    return DetectedFace(location=(0, 1, 1, 0), name=nom, confidence=0.9, is_known=True)


def _inconnu():
    return DetectedFace(location=(0, 1, 1, 0), name=UNKNOWN_NAME, confidence=0.2, is_known=False)


def _profil(**kw):
    base = {"name": "test", "label": "Test"}
    base.update(kw)
    return SurveillanceProfile(**base)


def test_alerte_sur_inconnu_quand_active():
    assert _profil(alert_on_unknown=True).should_alert([_inconnu()]) is True


def test_pas_d_alerte_sur_inconnu_quand_desactive():
    assert _profil(alert_on_unknown=False).should_alert([_inconnu()]) is False


def test_pas_d_alerte_sur_connu_par_defaut():
    assert _profil(alert_on_unknown=True).should_alert([_connu()]) is False


def test_alerte_sur_connu_quand_active():
    assert _profil(alert_on_known=True).should_alert([_connu()]) is True


def test_personne_ciblee_declenche_meme_si_alertes_connues_desactivees():
    p = _profil(alert_on_unknown=False, alert_on_known=False, target_persons=["Bob"])
    assert p.should_alert([_connu("Bob")]) is True


def test_personne_non_ciblee_ne_declenche_pas():
    p = _profil(alert_on_unknown=False, alert_on_known=False, target_persons=["Bob"])
    assert p.should_alert([_connu("Alice")]) is False


def test_aucune_alerte_sans_visage():
    assert _profil(alert_on_unknown=True, alert_on_known=True).should_alert([]) is False


def test_should_record_suit_le_champ():
    assert _profil(record_video=True).should_record() is True
    assert _profil(record_video=False).should_record() is False


def test_profils_par_defaut_presents():
    assert set(DEFAULT_PROFILES) == {"present", "absent", "nuit"}


def test_profil_present_n_enregistre_pas_et_n_alerte_pas():
    p = DEFAULT_PROFILES["present"]
    assert p.should_record() is False
    assert p.should_alert([_inconnu()]) is False


def test_profil_absent_enregistre_et_alerte_sur_inconnu():
    p = DEFAULT_PROFILES["absent"]
    assert p.should_record() is True
    assert p.should_alert([_inconnu()]) is True


def test_profil_nuit_analyse_en_continu():
    assert DEFAULT_PROFILES["nuit"].motion_required is False


def test_aller_retour_dict():
    p = _profil(target_persons=["Bob"], record_video=True)
    assert SurveillanceProfile.from_dict(p.to_dict()) == p


def test_from_dict_ignore_enabled_camera_uids_obsolete():
    """profiles.json existant contient ce champ, desormais supprime."""
    p = SurveillanceProfile.from_dict({"name": "x", "label": "X", "enabled_camera_uids": ["abc"]})
    assert p.name == "x"
    assert not hasattr(p, "enabled_camera_uids")


def test_seuil_de_reconnaissance_invalide_rejete():
    with pytest.raises(ValueError, match="recognition_threshold"):
        _profil(recognition_threshold=1.5)


def test_intervalle_d_analyse_negatif_rejete():
    with pytest.raises(ValueError, match="analysis_interval"):
        _profil(analysis_interval=-1.0)
