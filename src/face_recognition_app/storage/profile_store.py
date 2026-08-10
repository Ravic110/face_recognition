"""
profile_store.py
ADAPTATEUR TEMPORAIRE — conserve l'API utilisée par ui/surveillance_dashboard.py.

Délègue à ProfileRepository. Sera supprimé en Phase 3.
"""

from __future__ import annotations

from ..domain.profile import DEFAULT_PROFILES, SurveillanceProfile
from ..settings import AppSettings
from .profile_repository import ProfileRepository

__all__ = ["DEFAULT_PROFILES", "ProfileStore", "SurveillanceProfile"]


class ProfileStore:
    """Façade de compatibilité au-dessus de ProfileRepository."""

    def __init__(self, profiles_file=None) -> None:
        self._repo = ProfileRepository(AppSettings.create())

    @property
    def repository(self) -> ProfileRepository:
        return self._repo

    def list_profiles(self):
        return self._repo.list_all()

    def get(self, name):
        return self._repo.get(name)

    def get_active(self):
        return self._repo.get_active()

    @property
    def active_name(self):
        return self._repo.active_name

    def set_active(self, name):
        return self._repo.set_active(name)

    def save_profile(self, profile):
        self._repo.save_profile(profile)

    def delete_profile(self, name):
        return self._repo.delete_profile(name)

    def save(self):
        self._repo.save()
