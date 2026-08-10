from datetime import datetime

from face_recognition_app.domain.detection import (
    UNKNOWN_NAME,
    DetectedFace,
    SurveillanceEvent,
)


def _connu(nom="Alice"):
    return DetectedFace(location=(0, 10, 10, 0), name=nom, confidence=0.9, is_known=True)


def _inconnu():
    return DetectedFace(location=(0, 10, 10, 0), name=UNKNOWN_NAME, confidence=0.2, is_known=False)


def test_aller_retour_dict_visage():
    f = _connu()
    assert DetectedFace.from_dict(f.to_dict()) == f


def test_from_dict_tolere_une_localisation_absente():
    """Les evenements deja en base ne stockent pas la localisation."""
    f = DetectedFace.from_dict({"name": "Alice", "confidence": 0.9, "is_known": True})
    assert f.name == "Alice"
    assert f.location == (0, 0, 0, 0)


def test_known_names_ne_liste_que_les_connus():
    evt = SurveillanceEvent("uid", "Salon", 1000.0, [_connu("Alice"), _inconnu()])
    assert evt.known_names == ["Alice"]


def test_unknown_count():
    evt = SurveillanceEvent("uid", "Salon", 1000.0, [_connu(), _inconnu(), _inconnu()])
    assert evt.unknown_count == 2
    assert evt.has_unknown is True


def test_has_unknown_faux_si_tous_connus():
    evt = SurveillanceEvent("uid", "Salon", 1000.0, [_connu("Alice"), _connu("Bob")])
    assert evt.has_unknown is False
    assert evt.unknown_count == 0


def test_evenement_sans_visage():
    evt = SurveillanceEvent("uid", "Salon", 1000.0, [])
    assert evt.known_names == []
    assert evt.has_unknown is False


def test_dt_convertit_le_timestamp():
    ts = datetime(2026, 5, 7, 14, 30, 0).timestamp()
    assert SurveillanceEvent("uid", "Salon", ts, []).dt == datetime(2026, 5, 7, 14, 30, 0)
