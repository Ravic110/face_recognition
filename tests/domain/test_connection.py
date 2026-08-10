import pytest

from face_recognition_app.domain.connection import ConnectionState, ReconnectPolicy


def test_etats_disponibles():
    assert {s.name for s in ConnectionState} == {"DISCONNECTED", "CONNECTING", "CONNECTED"}


def test_premiere_tentative_utilise_le_delai_minimal():
    assert ReconnectPolicy(delay_min=2.0).next_delay(1) == pytest.approx(2.0)


def test_le_delai_double_a_chaque_tentative():
    p = ReconnectPolicy(delay_min=2.0, factor=2.0)
    assert [p.next_delay(n) for n in (1, 2, 3, 4)] == pytest.approx([2.0, 4.0, 8.0, 16.0])


def test_le_delai_est_plafonne():
    p = ReconnectPolicy(delay_min=2.0, delay_max=10.0, factor=2.0)
    assert p.next_delay(10) == pytest.approx(10.0)


def test_tentative_zero_ou_negative_donne_le_minimum():
    p = ReconnectPolicy(delay_min=2.0)
    assert p.next_delay(0) == pytest.approx(2.0)
    assert p.next_delay(-5) == pytest.approx(2.0)


def test_politique_est_pure_aucun_etat_mute():
    p = ReconnectPolicy(delay_min=2.0, factor=2.0)
    assert p.next_delay(3) == p.next_delay(3)


def test_delai_minimal_negatif_rejete():
    with pytest.raises(ValueError, match="delay_min"):
        ReconnectPolicy(delay_min=-1.0)


def test_maximum_inferieur_au_minimum_rejete():
    with pytest.raises(ValueError, match="delay_max"):
        ReconnectPolicy(delay_min=10.0, delay_max=5.0)


def test_facteur_inferieur_a_un_rejete():
    with pytest.raises(ValueError, match="factor"):
        ReconnectPolicy(factor=0.5)
