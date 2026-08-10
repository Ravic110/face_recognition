"""
connection.py
États de connexion d'une source vidéo et politique de reconnexion.

Le backoff était auparavant un attribut mutable mis à jour au fil de la boucle
de lecture, donc impossible à tester isolément. Il devient ici une fonction pure
de l'indice de tentative.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class ConnectionState(Enum):
    """
    États d'une source vidéo.

        DISCONNECTED ──tentative──► CONNECTING ──succès──► CONNECTED
              ▲                          │                     │
              └──────échec, backoff──────┘◄───perte de flux────┘

    `DISCONNECTED` retente toujours : c'est précisément ce qui manquait à
    l'implémentation précédente, qui restait inerte après un échec de
    réouverture.
    """

    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()


@dataclass(frozen=True)
class ReconnectPolicy:
    """Backoff exponentiel plafonné. `next_delay` est une fonction pure."""

    delay_min: float = 2.0
    delay_max: float = 60.0
    factor: float = 2.0

    def __post_init__(self) -> None:
        if self.delay_min <= 0:
            raise ValueError(f"delay_min doit être strictement positif, reçu {self.delay_min}")
        if self.delay_max < self.delay_min:
            raise ValueError(
                f"delay_max ({self.delay_max}) doit être au moins égal à "
                f"delay_min ({self.delay_min})"
            )
        if self.factor < 1.0:
            raise ValueError(f"factor doit être au moins 1.0, reçu {self.factor}")

    def next_delay(self, attempt: int) -> float:
        """Délai avant la n-ième tentative de reconnexion (n commence à 1)."""
        rang = max(0, attempt - 1)
        return min(self.delay_min * (self.factor**rang), self.delay_max)
