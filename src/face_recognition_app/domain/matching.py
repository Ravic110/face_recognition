"""
matching.py
Comparaison d'encodages faciaux — numpy pur.

Un encodage est un vecteur de 128 flottants. La distance entre deux visages est
la norme euclidienne de leur différence : deux encodages du même visage sont
proches, deux visages différents sont éloignés.

Ce module ne dépend ni de dlib ni d'OpenCV, ce qui le rend testable en
millisecondes et exécutable en CI sans compilation.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

ENCODING_LENGTH = 128


def face_distance(known: np.ndarray, unknown: np.ndarray) -> np.ndarray:
    """
    Distance euclidienne entre chaque encodage connu et l'encodage inconnu.

    Args:
        known: tableau (N, 128) des encodages de référence.
        unknown: vecteur (128,) à comparer.

    Returns:
        Tableau (N,) des distances. Vide si `known` est vide.
    """
    if known.size == 0:
        return np.empty(0, dtype=float)
    return np.linalg.norm(np.asarray(known, dtype=float) - np.asarray(unknown, dtype=float), axis=1)


def best_match(
    known: np.ndarray,
    unknown: np.ndarray,
    threshold: float,
) -> tuple[int | None, float]:
    """
    Trouve l'encodage connu le plus proche.

    Returns:
        (indice, distance) si la distance minimale est strictement sous le seuil,
        (None, distance) sinon. Sur une base vide : (None, inf).
    """
    distances = face_distance(known, unknown)
    if distances.size == 0:
        return None, float("inf")
    idx = int(np.argmin(distances))
    dist = float(distances[idx])
    return (idx if dist < threshold else None), dist


def find_duplicate(
    unknown: np.ndarray,
    existing: Sequence[tuple[str, np.ndarray]],
    tolerance: float,
) -> str | None:
    """
    Nom de la personne déjà enregistrée dont le visage correspond, ou None.

    Retourne la correspondance la plus proche — et non la première rencontrée,
    comme le faisait l'implémentation précédente.
    """
    if not existing:
        return None
    names = [name for name, _ in existing]
    known = np.array([enc for _, enc in existing], dtype=float)
    idx, _ = best_match(known, unknown, tolerance)
    return names[idx] if idx is not None else None
