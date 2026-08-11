import pytest

from face_recognition_app.domain.camera import CameraConfig


def test_uid_genere_automatiquement():
    c = CameraConfig(name="Salon", source_type="webcam", source=0)
    assert len(c.uid) == 8


def test_aller_retour_dict_webcam():
    c = CameraConfig(name="Salon", source_type="webcam", source=0)
    assert CameraConfig.from_dict(c.to_dict()) == c


def test_aller_retour_dict_camera_ip_avec_roi():
    c = CameraConfig(
        name="Couloir",
        source_type="ip",
        source="http://10.0.0.5:8080/video",
        roi=(10, 20, 300, 200),
    )
    restaure = CameraConfig.from_dict(c.to_dict())
    assert restaure == c
    assert restaure.roi == (10, 20, 300, 200)


def test_from_dict_lit_le_format_existant():
    """Le format de cameras.json actuel doit rester lisible."""
    data = {
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
    c = CameraConfig.from_dict(data)
    assert c.uid == "6dc1bce5"
    assert c.name == "camera couloir"
    assert c.roi is None
    assert c.is_ip is True


def test_from_dict_applique_les_defauts_sur_champs_absents():
    c = CameraConfig.from_dict({"name": "X", "source_type": "webcam", "source": 1})
    assert c.enabled is True
    assert c.width == 640
    assert c.height == 480
    assert c.detection_model == "hog"


def test_type_de_source_invalide_rejete():
    with pytest.raises(ValueError, match="source_type"):
        CameraConfig(name="X", source_type="satellite", source=0)


def test_modele_de_detection_invalide_rejete():
    with pytest.raises(ValueError, match="detection_model"):
        CameraConfig(name="X", source_type="webcam", source=0, detection_model="magique")


def test_is_ip_faux_pour_webcam():
    assert CameraConfig(name="X", source_type="webcam", source=0).is_ip is False


# ── Effet miroir ──────────────────────────────────────────────────────────────


def test_webcam_est_en_miroir_par_defaut():
    """Une webcam sert de miroir : on s'attend a se voir comme dans une glace."""
    assert CameraConfig(name="Salon", source_type="webcam", source=0).mirror is True


def test_camera_ip_n_est_pas_en_miroir_par_defaut():
    """Retourner une camera de surveillance rendrait la scene illisible."""
    c = CameraConfig(name="Couloir", source_type="ip", source="http://x/video")
    assert c.mirror is False


def test_miroir_explicite_prime_sur_le_defaut():
    assert CameraConfig(name="X", source_type="webcam", source=0, mirror=False).mirror is False
    assert CameraConfig(name="Y", source_type="ip", source="http://x", mirror=True).mirror is True


def test_miroir_persiste_dans_le_dict():
    c = CameraConfig(name="X", source_type="webcam", source=0)
    assert c.to_dict()["mirror"] is True
    assert CameraConfig.from_dict(c.to_dict()).mirror is True


def test_cameras_json_existant_recoit_le_defaut_par_type():
    """Les entrees deja enregistrees n'ont pas la cle mirror."""
    webcam = CameraConfig.from_dict({"name": "W", "source_type": "webcam", "source": 0})
    ip = CameraConfig.from_dict({"name": "I", "source_type": "ip", "source": "http://x"})
    assert webcam.mirror is True
    assert ip.mirror is False
