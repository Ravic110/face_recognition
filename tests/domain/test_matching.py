import numpy as np
import pytest

from face_recognition_app.domain.matching import best_match, face_distance, find_duplicate


def test_face_distance_identique_vaut_zero():
    known = np.zeros((1, 128))
    unknown = np.zeros(128)
    assert face_distance(known, unknown)[0] == pytest.approx(0.0)


def test_face_distance_calcule_la_norme_euclidienne():
    known = np.array([[3.0] + [0.0] * 127])
    unknown = np.zeros(128)
    assert face_distance(known, unknown)[0] == pytest.approx(3.0)


def test_face_distance_sur_base_vide_retourne_tableau_vide():
    assert face_distance(np.empty((0, 128)), np.zeros(128)).size == 0


def test_best_match_retourne_le_plus_proche_pas_le_premier():
    known = np.array([[0.4] + [0.0] * 127, [0.1] + [0.0] * 127])
    idx, dist = best_match(known, np.zeros(128), threshold=0.5)
    assert idx == 1
    assert dist == pytest.approx(0.1)


def test_best_match_au_dela_du_seuil_retourne_none():
    known = np.array([[0.9] + [0.0] * 127])
    idx, dist = best_match(known, np.zeros(128), threshold=0.5)
    assert idx is None
    assert dist == pytest.approx(0.9)


def test_best_match_base_vide():
    idx, dist = best_match(np.empty((0, 128)), np.zeros(128), threshold=0.5)
    assert idx is None
    assert dist == float("inf")


def test_find_duplicate_identique():
    vec = np.full(128, 0.1)
    assert find_duplicate(vec, [("Alice", vec)], tolerance=0.6) == "Alice"


def test_find_duplicate_aucune_correspondance():
    assert find_duplicate(np.zeros(128), [("Alice", np.ones(128))], tolerance=0.6) is None


def test_find_duplicate_base_vide():
    assert find_duplicate(np.zeros(128), [], tolerance=0.6) is None


def test_find_duplicate_respecte_une_tolerance_stricte():
    autre = np.zeros(128)
    autre[0] = 0.3
    assert find_duplicate(np.zeros(128), [("Alice", autre)], tolerance=0.01) is None


def test_find_duplicate_retourne_le_plus_proche():
    proche = np.zeros(128)
    proche[0] = 0.05
    loin = np.zeros(128)
    loin[0] = 0.4
    existing = [("Loin", loin), ("Proche", proche)]
    assert find_duplicate(np.zeros(128), existing, tolerance=0.6) == "Proche"
