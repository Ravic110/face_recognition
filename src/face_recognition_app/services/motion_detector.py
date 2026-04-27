"""
motion_detector.py
Détection de mouvement par différence de frames (fond adaptatif).

Principe :
  - Maintient un fond adaptatif via MOG2 (Gaussian Mixture)
  - Retourne True si le score de mouvement dépasse un seuil configurable
  - Fournit un masque de mouvement pour délimiter les zones actives

Utilisé par SurveillanceEngine pour n'activer la reconnaissance faciale
que lorsqu'un mouvement est détecté → économie CPU significative.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class MotionDetector:
    """
    Détecteur de mouvement basé sur la soustraction de fond (MOG2).

    Args:
        sensitivity  : surface minimale d'un contour en pixels² (plus bas = plus sensible)
        min_area_ratio : fraction minimale de l'image couverte par le mouvement (0.0–1.0)
        roi          : zone d'intérêt (x, y, w, h) en pixels, ou None pour toute l'image
    """

    def __init__(
        self,
        sensitivity: int = 500,
        min_area_ratio: float = 0.002,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        self._sensitivity = sensitivity
        self._min_area_ratio = min_area_ratio
        self._roi = roi
        # MOG2 : fond adaptatif robuste aux changements lents (luminosité…)
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=50,
            detectShadows=False,
        )
        self._frame_count = 0

    # ── API publique ──────────────────────────────────────────────────────────

    def update(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Analyse une frame BGR et retourne (mouvement_détecté, score).

        score : fraction de l'image en mouvement (0.0 → 1.0)
        """
        region = self._apply_roi(frame)
        gray = cv2.GaussianBlur(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY), (21, 21), 0)
        mask = self._subtractor.apply(gray)

        # Morphologie pour supprimer le bruit
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Score = proportion de pixels en mouvement
        h, w = mask.shape
        total_px = h * w
        motion_px = int(np.count_nonzero(mask))
        score = motion_px / total_px if total_px > 0 else 0.0

        # Vérification par contours (filtre les faux-positifs ponctuels)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant = any(cv2.contourArea(c) > self._sensitivity for c in contours)

        detected = significant and score >= self._min_area_ratio
        self._frame_count += 1
        return detected, round(score, 4)

    def get_motion_mask(self, frame: np.ndarray) -> np.ndarray:
        """Retourne le masque binaire de mouvement sur la frame complète."""
        region = self._apply_roi(frame)
        gray = cv2.GaussianBlur(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY), (21, 21), 0)
        return self._subtractor.apply(gray)

    def set_roi(self, roi: Optional[Tuple[int, int, int, int]]) -> None:
        """Met à jour la zone d'intérêt (x, y, w, h)."""
        self._roi = roi
        # Réinitialiser le fond lors du changement de ROI
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=50, detectShadows=False)

    def reset(self) -> None:
        """Réinitialise le modèle de fond (ex. après un changement de scène)."""
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=50, detectShadows=False)
        self._frame_count = 0

    # ── Utilitaire ROI ────────────────────────────────────────────────────────

    def _apply_roi(self, frame: np.ndarray) -> np.ndarray:
        if self._roi is None:
            return frame
        x, y, w, h = self._roi
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)
        if x2 <= x1 or y2 <= y1:
            return frame
        return frame[y1:y2, x1:x2]
