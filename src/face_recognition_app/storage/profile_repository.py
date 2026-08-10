"""
profile_repository.py
Persistance des profils de surveillance dans `profiles.json`.

Format :
    {"active": "present", "profiles": [ {...}, ... ]}

Contient également la migration des champs d'alerte : ils vivaient auparavant
dans `alerts_config.json` en doublon du profil, et seule la version
`alerts_config.json` avait un effet. `migrate_alert_fields()` les importe dans
les profils puis les retire du fichier d'alertes, une seule fois.
"""

from __future__ import annotations

import json
import logging

from ..domain.profile import DEFAULT_PROFILES, SurveillanceProfile
from ..settings import AppSettings

logger = logging.getLogger(__name__)

# Champs d'alerte déplacés d'AlertConfig vers SurveillanceProfile
_CHAMPS_MIGRES = ("alert_on_unknown", "alert_on_known", "target_persons")


class ProfileRepository:
    """Charge, persiste et sélectionne les profils de surveillance."""

    FALLBACK = "present"

    def __init__(self, settings: AppSettings) -> None:
        self._file = settings.profiles_file
        self._alerts_file = settings.alerts_file
        self._profiles: dict[str, SurveillanceProfile] = {
            name: SurveillanceProfile.from_dict(p.to_dict()) for name, p in DEFAULT_PROFILES.items()
        }
        self._active = self.FALLBACK
        self._load()

    # ── Persistance ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._file.exists():
            return
        try:
            data = json.loads(self._file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("profiles.json illisible, profils par défaut utilisés : %s", exc)
            return

        for brut in data.get("profiles", []):
            try:
                profil = SurveillanceProfile.from_dict(brut)
            except (TypeError, ValueError) as exc:
                logger.error("Profil invalide ignoré (%s) : %s", brut.get("name", "?"), exc)
                continue
            self._profiles[profil.name] = profil

        demande = data.get("active", self.FALLBACK)
        self._active = demande if demande in self._profiles else self.FALLBACK

    def save(self) -> None:
        data = {
            "active": self._active,
            "profiles": [p.to_dict() for p in self._profiles.values()],
        }
        self._file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Lecture ───────────────────────────────────────────────────────────────

    def list_all(self) -> list[SurveillanceProfile]:
        return list(self._profiles.values())

    def get(self, name: str) -> SurveillanceProfile | None:
        return self._profiles.get(name)

    def get_active(self) -> SurveillanceProfile:
        return self._profiles.get(self._active, DEFAULT_PROFILES[self.FALLBACK])

    @property
    def active_name(self) -> str:
        return self._active

    # ── Écriture ──────────────────────────────────────────────────────────────

    def set_active(self, name: str) -> bool:
        if name not in self._profiles:
            logger.warning("Profil inconnu demandé : %s", name)
            return False
        self._active = name
        self.save()
        logger.info("Profil actif : %s", name)
        return True

    def save_profile(self, profile: SurveillanceProfile) -> None:
        self._profiles[profile.name] = profile
        self.save()

    def delete_profile(self, name: str) -> bool:
        if name in DEFAULT_PROFILES:
            logger.warning("Suppression refusée : '%s' est un profil par défaut", name)
            return False
        if name not in self._profiles:
            return False
        if name == self._active:
            self._active = self.FALLBACK
        del self._profiles[name]
        self.save()
        return True

    # ── Migration ─────────────────────────────────────────────────────────────

    def migrate_alert_fields(self) -> bool:
        """
        Importe les champs d'alerte d'`alerts_config.json` dans tous les profils,
        puis les retire du fichier d'alertes.

        Returns:
            True si une migration a eu lieu, False s'il n'y avait rien à migrer.
        """
        if not self._alerts_file.exists():
            return False
        try:
            data = json.loads(self._alerts_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("alerts_config.json illisible, migration ignorée : %s", exc)
            return False

        presents = {k: data[k] for k in _CHAMPS_MIGRES if k in data}
        if not presents:
            return False

        for profil in self._profiles.values():
            for champ, valeur in presents.items():
                setattr(profil, champ, valeur)
        self.save()

        for champ in presents:
            data.pop(champ)
        self._alerts_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        logger.info(
            "Migration : champs d'alerte %s déplacés d'alerts_config.json vers les profils",
            ", ".join(sorted(presents)),
        )
        return True
