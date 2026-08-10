# Refactorisation — Phase 2A : Fiabilité caméra et sécurité de l'API — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rendre le système sûr à exposer et fiable dans la durée — une caméra qui perd son flux se reconnecte toujours, et l'API ne répond plus à personne sans authentification.

**Architecture:** Deux chantiers indépendants. Le premier remplace la boucle de reconnexion sans état de `CameraSource` par une machine à états explicite pilotée par une politique de backoff pure et testable. Le second réécrit `ApiServer` autour de `werkzeug.serving.make_server` avec authentification par clé, écoute sur `127.0.0.1` par défaut et séparation lecture/écriture.

**Tech Stack:** Python 3.12, OpenCV (capture), Flask + Werkzeug (API), `hmac`/`secrets` (authentification), pytest.

## Global Constraints

- Reprend les contraintes de la Phase 1 : `domain/` n'importe ni `face_recognition`, ni `dlib`, ni `cv2`, ni `tkinter` ; aucun module ne lit de configuration à l'import ; aucun `except Exception: pass` nu.
- **Aucun test n'ouvre de caméra réelle, ne lie de port réseau, ni n'exige un écran.** La capture est simulée en surchargeant `_open_capture()` et `_read_raw()` ; l'API est testée via `app.test_client()`.
- **Aucune vérification ne s'exécute contre les données réelles du projet.** Tout ce qui construit un `AppSettings` en dehors des tests utilise une copie temporaire (`mktemp -d`).
- Tous les `time.sleep` des boucles de service deviennent des `Event.wait()`, afin que `stop()` rende la main immédiatement.
- La suite complète doit rester sous les 10 secondes.
- Tout commit laisse `ruff check`, `ruff format --check`, `mypy` et `pytest` verts, et `python main.py` fonctionnel.
- Messages de commit en français, préfixés `feat:`, `fix:`, `refactor:`, `test:` ou `chore:`.

## Structure des fichiers

| Fichier | Responsabilité |
|---|---|
| `domain/connection.py` (créé) | `ConnectionState`, `ReconnectPolicy` — calcul de backoff pur, sans I/O |
| `services/camera_source.py` (réécrit) | Machine à états de connexion, boucle de lecture interruptible |
| `api/__init__.py` (créé) | Paquet |
| `api/auth.py` (créé) | Résolution de la clé, comparaison en temps constant, limitation des tentatives |
| `api/server.py` (créé) | Serveur Werkzeug arrêtable, routes, séparation lecture/écriture |
| `services/api_server.py` (supprimé) | Remplacé par `api/server.py` |
| `ui/surveillance_dashboard.py` (modifié) | Câblage sur le nouveau serveur, affichage de la clé |

---

### Task 1: `domain/connection.py` — états et politique de reconnexion

**Files:**
- Create: `src/face_recognition_app/domain/connection.py`
- Create: `tests/domain/test_connection.py`

**Interfaces:**
- Consumes: rien
- Produces:
  - `ConnectionState` (`Enum`) avec les membres `DISCONNECTED`, `CONNECTING`, `CONNECTED`
  - `ReconnectPolicy(delay_min: float = 2.0, delay_max: float = 60.0, factor: float = 2.0)` avec `next_delay(attempt: int) -> float`

**Pourquoi :** le backoff est aujourd'hui un état mutable (`_reconnect_delay`) mis à jour au fil de la boucle, ce qui le rend intestable isolément. En faire une fonction pure de l'indice de tentative permet de le vérifier en microsecondes et d'injecter des délais minuscules dans les tests de la tâche 2.

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/domain/test_connection.py` :

```python
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
```

- [ ] **Step 2: Vérifier que les tests échouent**

Run: `.venv/bin/python -m pytest tests/domain/test_connection.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'face_recognition_app.domain.connection'`

- [ ] **Step 3: Implémenter**

Créer `src/face_recognition_app/domain/connection.py` :

```python
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
```

- [ ] **Step 4: Vérifier que les tests passent**

Run: `.venv/bin/python -m pytest tests/domain/test_connection.py`
Expected: `9 passed`

- [ ] **Step 5: Vérifier la qualité et commiter**

Run: `.venv/bin/ruff check src tests && .venv/bin/ruff format src tests && .venv/bin/mypy`
Expected: `All checks passed!` puis `Success: no issues found`

```bash
git add src/face_recognition_app/domain/connection.py tests/domain/test_connection.py
git commit -m "feat: domain/connection.py, etats et backoff de reconnexion

Le backoff etait un attribut mutable mis a jour dans la boucle de
lecture, donc intestable isolement. Il devient une fonction pure de
l'indice de tentative."
```

---

### Task 2: `CameraSource` — machine à états et boucle interruptible (bug ③)

**Files:**
- Modify: `src/face_recognition_app/services/camera_source.py`
- Create: `tests/services/test_camera_source.py`

**Interfaces:**
- Consumes: `ConnectionState`, `ReconnectPolicy` (Task 1), `CameraConfig` (Phase 1)
- Produces sur `CameraSource` :
  - `__init__(config: CameraConfig, policy: ReconnectPolicy | None = None)`
  - `start() -> bool`, `stop() -> None` (idempotents)
  - `get_frame() -> np.ndarray | None`
  - propriétés `state: ConnectionState`, `is_connected: bool`, `is_running: bool`, `reconnect_attempts: int`, `next_retry_in: float`
  - points de surcharge pour les tests : `_open_capture() -> bool`, `_read_raw() -> tuple[bool, np.ndarray | None]`, `_release() -> None`

**Le défaut corrigé :** dans `_read_loop`, après une lecture ratée, `_release()` met `_cap` à `None` puis `_open_capture()` est tentée. Si cette réouverture échoue, `_cap` reste `None` et l'itération suivante prend la branche `else: time.sleep(0.1)` — définitivement. La caméra ne se reconnecte plus jamais. C'est le scénario nominal d'une caméra IP.

Trois défauts secondaires au même endroit : `time.sleep(delay)` peut durer 60 s et ignore `_running`, donc `stop()` (join à 3 s) abandonne le thread qui manipule ensuite une capture libérée ; le `cap` d'une ouverture ratée n'est jamais `release()` ; et `start()` n'est pas idempotent.

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/services/test_camera_source.py` :

```python
import threading
import time

import numpy as np
import pytest

from face_recognition_app.domain.camera import CameraConfig
from face_recognition_app.domain.connection import ConnectionState, ReconnectPolicy
from face_recognition_app.services.camera_source import CameraSource

# Backoff minuscule : les tests ne doivent jamais attendre réellement.
POLITIQUE_RAPIDE = ReconnectPolicy(delay_min=0.01, delay_max=0.02, factor=2.0)


def _config():
    return CameraConfig(name="Test", source_type="webcam", source=0)


class SourceSimulee(CameraSource):
    """
    Source contrôlée par le test — aucune caméra réelle.

    `ouvertures_echouees` : nombre d'ouvertures qui échouent avant de réussir.
    `lectures_avant_perte` : nombre de lectures réussies avant une perte de flux.
    """

    def __init__(self, config, policy=None, ouvertures_echouees=0, lectures_avant_perte=None):
        super().__init__(config, policy=policy)
        self.ouvertures_echouees = ouvertures_echouees
        self.lectures_avant_perte = lectures_avant_perte
        self.ouvertures_tentees = 0
        self.liberations = 0
        self._lectures = 0
        self._ouvert = False

    def _open_capture(self) -> bool:
        self.ouvertures_tentees += 1
        if self.ouvertures_tentees <= self.ouvertures_echouees:
            return False
        self._ouvert = True
        self._lectures = 0
        return True

    def _read_raw(self):
        if not self._ouvert:
            return False, None
        if self.lectures_avant_perte is not None and self._lectures >= self.lectures_avant_perte:
            return False, None
        self._lectures += 1
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def _release(self) -> None:
        self.liberations += 1
        self._ouvert = False


def _attendre(predicat, timeout=2.0):
    """Attend qu'une condition devienne vraie, sans dormir inutilement."""
    limite = time.monotonic() + timeout
    while time.monotonic() < limite:
        if predicat():
            return True
        time.sleep(0.005)
    return False


def test_demarre_et_fournit_une_frame():
    src = SourceSimulee(_config(), policy=POLITIQUE_RAPIDE)
    assert src.start() is True
    try:
        assert _attendre(lambda: src.get_frame() is not None)
        assert src.state is ConnectionState.CONNECTED
        assert src.is_connected is True
    finally:
        src.stop()


def test_echec_d_ouverture_initiale():
    src = SourceSimulee(_config(), policy=POLITIQUE_RAPIDE, ouvertures_echouees=99)
    assert src.start() is False
    assert src.state is ConnectionState.DISCONNECTED
    assert src.is_running is False


def test_reconnexion_apres_perte_de_flux():
    src = SourceSimulee(_config(), policy=POLITIQUE_RAPIDE, lectures_avant_perte=2)
    src.start()
    try:
        assert _attendre(lambda: src.ouvertures_tentees >= 3)
        assert src.state is ConnectionState.CONNECTED
    finally:
        src.stop()


def test_reconnexion_persiste_apres_plusieurs_echecs():
    """
    Le defaut historique : apres un echec de reouverture, la boucle restait
    inerte et la camera ne revenait jamais.
    """
    src = SourceSimulee(_config(), policy=POLITIQUE_RAPIDE, lectures_avant_perte=1)
    src.start()
    try:
        # La source perd le flux en boucle ; ce qui compte est qu'elle retente
        # indefiniment au lieu de s'arreter apres le premier echec.
        assert _attendre(lambda: src.ouvertures_tentees >= 5, timeout=3.0)
    finally:
        src.stop()


def test_ouverture_ratee_libere_la_capture():
    """Une capture ouverte mais inutilisable ne doit pas fuir."""
    src = SourceSimulee(_config(), policy=POLITIQUE_RAPIDE, lectures_avant_perte=0)
    src.start()
    try:
        assert _attendre(lambda: src.liberations >= 1)
    finally:
        src.stop()


def test_stop_rend_la_main_pendant_l_attente_de_reconnexion():
    """Le sleep etait bloquant jusqu'a 60s ; stop() doit interrompre l'attente."""
    lente = ReconnectPolicy(delay_min=30.0, delay_max=60.0)
    src = SourceSimulee(_config(), policy=lente, lectures_avant_perte=1, ouvertures_echouees=0)
    src.start()
    try:
        assert _attendre(lambda: src.state is not ConnectionState.CONNECTED, timeout=2.0)
    finally:
        debut = time.monotonic()
        src.stop()
        assert time.monotonic() - debut < 2.0


def test_stop_est_idempotent():
    src = SourceSimulee(_config(), policy=POLITIQUE_RAPIDE)
    src.start()
    src.stop()
    src.stop()
    assert src.is_running is False


def test_start_est_idempotent():
    src = SourceSimulee(_config(), policy=POLITIQUE_RAPIDE)
    try:
        assert src.start() is True
        assert src.start() is True
        assert _attendre(lambda: src.get_frame() is not None)
        assert threading.active_count() < 20
    finally:
        src.stop()


def test_le_compteur_de_tentatives_se_reinitialise_apres_succes():
    src = SourceSimulee(_config(), policy=POLITIQUE_RAPIDE, lectures_avant_perte=3)
    src.start()
    try:
        assert _attendre(lambda: src.ouvertures_tentees >= 2)
        assert _attendre(lambda: src.reconnect_attempts == 0)
    finally:
        src.stop()


def test_next_retry_in_est_nul_quand_connecte():
    src = SourceSimulee(_config(), policy=POLITIQUE_RAPIDE)
    src.start()
    try:
        assert _attendre(lambda: src.state is ConnectionState.CONNECTED)
        assert src.next_retry_in == pytest.approx(0.0)
    finally:
        src.stop()


def test_get_frame_retourne_une_copie():
    src = SourceSimulee(_config(), policy=POLITIQUE_RAPIDE)
    src.start()
    try:
        assert _attendre(lambda: src.get_frame() is not None)
        a, b = src.get_frame(), src.get_frame()
        assert a is not b
    finally:
        src.stop()
```

- [ ] **Step 2: Vérifier que les tests échouent**

Run: `.venv/bin/python -m pytest tests/services/test_camera_source.py`
Expected: FAIL — `TypeError: CameraSource.__init__() got an unexpected keyword argument 'policy'`

- [ ] **Step 3: Réécrire `CameraSource`**

Remplacer intégralement le contenu de `src/face_recognition_app/services/camera_source.py` situé **après** le bloc d'imports (c'est-à-dire tout à partir du commentaire `# ── Classe de base ──`) par :

```python
# ── Classe de base ────────────────────────────────────────────────────────────


class CameraSource(ABC):
    """
    Source vidéo générique avec boucle de lecture en arrière-plan.

    Machine à états :

        DISCONNECTED ──tentative──► CONNECTING ──succès──► CONNECTED
              ▲                          │                     │
              └──────échec, backoff──────┘◄───perte de flux────┘

    `DISCONNECTED` retente toujours. L'implémentation précédente restait
    définitivement inerte lorsqu'une réouverture échouait.

    Les attentes utilisent `Event.wait()` et non `time.sleep()`, afin que
    `stop()` rende la main immédiatement même au milieu d'un backoff de 60 s.

    Usage :
        src = WebcamSource(config)
        src.start()
        frame = src.get_frame()   # None si pas encore de frame
        src.stop()
    """

    # Pause de la boucle quand la source est saine, pour ne pas saturer le CPU
    IDLE_POLL = 0.005

    def __init__(self, config: CameraConfig, policy: ReconnectPolicy | None = None) -> None:
        self.config = config
        self._policy = policy or ReconnectPolicy()
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._state = ConnectionState.DISCONNECTED
        self._attempts = 0
        self._next_retry_in = 0.0

    # ── Propriétés ────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def uid(self) -> str:
        return self.config.uid

    @property
    def state(self) -> ConnectionState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_connected(self) -> bool:
        return self._state is ConnectionState.CONNECTED

    @property
    def reconnect_attempts(self) -> int:
        """Nombre d'échecs consécutifs. Remis à zéro dès qu'une lecture réussit."""
        return self._attempts

    @property
    def next_retry_in(self) -> float:
        """Délai avant la prochaine tentative, 0 si la source est connectée."""
        return self._next_retry_in

    # ── Cycle de vie ──────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Ouvre la capture et démarre la boucle de lecture. Idempotent."""
        if self.is_running:
            return True

        self._stop_event.clear()
        self._state = ConnectionState.CONNECTING
        if not self._try_open():
            self._state = ConnectionState.DISCONNECTED
            logger.error("[%s] Impossible d'ouvrir la source : %s", self.name, self.config.source)
            return False

        self._thread = threading.Thread(
            target=self._read_loop,
            daemon=True,
            name=f"cam-{self.uid}",
        )
        self._thread.start()
        logger.info("[%s] Démarré (source=%s)", self.name, self.config.source)
        return True

    def stop(self) -> None:
        """Arrête la capture. Idempotent, et interrompt un backoff en cours."""
        self._stop_event.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=5.0)
            if thread.is_alive():
                logger.warning("[%s] Le thread de lecture ne s'est pas arrêté", self.name)
        self._release()
        self._state = ConnectionState.DISCONNECTED
        self._next_retry_in = 0.0
        logger.info("[%s] Arrêté", self.name)

    # ── Lecture de frame ──────────────────────────────────────────────────────

    def get_frame(self) -> np.ndarray | None:
        """Dernière frame disponible (copie, thread-safe), ou None."""
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    # ── Points de surcharge ───────────────────────────────────────────────────

    def _open_capture(self) -> bool:
        """Ouvre la capture. Retourne True si elle est utilisable."""
        cap = cv2.VideoCapture(self.config.source)
        if not cap.isOpened():
            # Libérer même une capture inutilisable : sinon le descripteur fuit.
            cap.release()
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self._cap = cap
        return True

    def _read_raw(self) -> tuple[bool, np.ndarray | None]:
        """Lit une frame. Retourne (succès, frame)."""
        if self._cap is None:
            return False, None
        ok, frame = self._cap.read()
        return bool(ok), frame

    def _release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ── Machine à états ───────────────────────────────────────────────────────

    def _try_open(self) -> bool:
        """Tente une ouverture et met l'état à jour."""
        if self._open_capture():
            self._state = ConnectionState.CONNECTED
            self._attempts = 0
            self._next_retry_in = 0.0
            return True
        self._release()
        return False

    def _handle_failure(self) -> None:
        """Perte de flux ou ouverture ratée : libère, attend, puis retente."""
        self._state = ConnectionState.DISCONNECTED
        self._release()
        self._attempts += 1
        delai = self._policy.next_delay(self._attempts)
        self._next_retry_in = delai
        logger.warning(
            "[%s] Flux indisponible (tentative %d), nouvelle connexion dans %.0f s…",
            self.name,
            self._attempts,
            delai,
        )
        # Event.wait plutôt que time.sleep : stop() interrompt l'attente.
        if self._stop_event.wait(delai):
            return
        self._state = ConnectionState.CONNECTING
        self._try_open()

    def _read_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._state is not ConnectionState.CONNECTED:
                # Toujours retenter — c'est ce qui manquait auparavant.
                self._handle_failure()
                continue

            ok, frame = self._read_raw()
            if not ok:
                self._handle_failure()
                continue

            with self._lock:
                self._latest_frame = frame
            self._stop_event.wait(self.IDLE_POLL)
```

- [ ] **Step 4: Adapter le bloc d'imports**

Dans le même fichier, remplacer :

```python
from ..domain.camera import CameraConfig

# CameraConfig est réexporté : l'UI l'importe depuis ce module.
__all__ = [
    "CameraConfig",
    "CameraSource",
    "IPCameraSource",
    "WebcamSource",
    "create_camera_source",
]
```

par :

```python
from ..domain.camera import CameraConfig
from ..domain.connection import ConnectionState, ReconnectPolicy

# CameraConfig est réexporté : l'UI l'importe depuis ce module.
__all__ = [
    "CameraConfig",
    "CameraSource",
    "ConnectionState",
    "IPCameraSource",
    "ReconnectPolicy",
    "WebcamSource",
    "create_camera_source",
]
```

Puis supprimer l'import devenu inutile `import time` s'il ne reste plus aucun `time.` dans le fichier.

- [ ] **Step 5: Adapter `WebcamSource`**

Plus bas dans le même fichier, `WebcamSource._open_capture` appelle `super()._open_capture()`. Le corps reste valide, mais vérifier qu'il correspond exactement à :

```python
class WebcamSource(CameraSource):
    """
    Caméra locale (USB ou intégrée).

    config.source = index entier (0, 1, 2…)
    """

    def _open_capture(self) -> bool:
        # Forcer l'index entier
        source = self.config.source
        self.config.source = int(source) if not isinstance(source, int) else source
        return super()._open_capture()
```

- [ ] **Step 6: Vérifier que les tests passent**

Run: `.venv/bin/python -m pytest tests/services/test_camera_source.py -x`
Expected: `11 passed`

- [ ] **Step 7: Vérifier que le `B024` du fichier peut être levé**

`CameraSource` hérite d'`ABC` sans méthode abstraite, ce que ruff signale par `B024`. Comme les sous-classes ne redéfinissent qu'optionnellement `_open_capture`, `ABC` n'apporte rien : le retirer.

Dans le fichier, remplacer `class CameraSource(ABC):` par `class CameraSource:` et supprimer `from abc import ABC` du bloc d'imports.

Puis retirer de `pyproject.toml` la ligne devenue caduque :

```
"src/face_recognition_app/services/camera_source.py" = ["B024"]
```

- [ ] **Step 8: Vérifier l'ensemble**

Run: `.venv/bin/ruff check src tests && .venv/bin/ruff format src tests && .venv/bin/mypy && .venv/bin/python -m pytest`
Expected: `All checks passed!`, `Success: no issues found`, `131 passed`

- [ ] **Step 9: Vérifier qu'aucune caméra réelle n'est ouverte par les tests**

Run: `.venv/bin/python -m pytest tests/services/test_camera_source.py --durations=3`
Expected: aucune durée supérieure à 3 s — si un test dure plus, c'est qu'il attend un vrai backoff.

- [ ] **Step 10: Commit**

```bash
git add -A src/face_recognition_app/services/camera_source.py tests/services pyproject.toml
git commit -m "fix: reconnexion camera par machine a etats (bug 3)

Apres une lecture ratee, _release() mettait _cap a None puis tentait une
reouverture. Si celle-ci echouait, _cap restait None et la boucle prenait
definitivement la branche inerte : la camera ne revenait jamais. C'est le
scenario nominal d'une camera IP.

L'etat DISCONNECTED retente desormais toujours. Les attentes passent de
time.sleep a Event.wait, donc stop() interrompt un backoff de 60s au lieu
d'abandonner un thread qui manipulait ensuite une capture liberee.

Corrige aussi : la capture d'une ouverture ratee n'etait jamais release,
et start() n'etait pas idempotent."
```

---

### Task 3: `api/auth.py` — clé, comparaison en temps constant, limitation

**Files:**
- Create: `src/face_recognition_app/api/__init__.py`
- Create: `src/face_recognition_app/api/auth.py`
- Create: `tests/api/test_auth.py`

**Interfaces:**
- Consumes: `AppSettings` (Phase 1 : `api_key_file`, `config_dir`)
- Produces:
  - `resolve_api_key(settings: AppSettings) -> str` — lit `FR_API_KEY`, sinon lit le fichier, sinon en génère une et l'écrit en `0600`
  - `ApiKeyGuard(api_key: str, max_attempts: int = 10, window_seconds: float = 300.0)` avec :
    - `check(provided: str | None, client_ip: str) -> bool`
    - `is_blocked(client_ip: str) -> bool`
    - `reset(client_ip: str) -> None`

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/api/test_auth.py` :

```python
import pytest

from face_recognition_app.api.auth import ApiKeyGuard, resolve_api_key
from face_recognition_app.settings import AppSettings


@pytest.fixture
def settings(tmp_path):
    s = AppSettings.create(project_root=tmp_path)
    s.ensure_directories()
    return s


# ── Résolution de la clé ──────────────────────────────────────────────────────


def test_cle_generee_au_premier_appel(settings):
    cle = resolve_api_key(settings)
    assert len(cle) >= 32
    assert settings.api_key_file.exists()


def test_cle_stable_entre_deux_appels(settings):
    assert resolve_api_key(settings) == resolve_api_key(settings)


def test_cle_fichier_en_permissions_restrictives(settings):
    resolve_api_key(settings)
    assert settings.api_key_file.stat().st_mode & 0o077 == 0


def test_variable_d_environnement_prioritaire(settings, monkeypatch):
    monkeypatch.setenv("FR_API_KEY", "cle-de-l-environnement")
    assert resolve_api_key(settings) == "cle-de-l-environnement"


def test_variable_d_environnement_n_ecrase_pas_le_fichier(settings, monkeypatch):
    depuis_fichier = resolve_api_key(settings)
    monkeypatch.setenv("FR_API_KEY", "temporaire")
    assert resolve_api_key(settings) == "temporaire"
    monkeypatch.delenv("FR_API_KEY")
    assert resolve_api_key(settings) == depuis_fichier


def test_fichier_vide_regenere_une_cle(settings):
    settings.api_key_file.write_text("   ")
    assert len(resolve_api_key(settings)) >= 32


# ── Vérification ──────────────────────────────────────────────────────────────


def test_bonne_cle_acceptee():
    assert ApiKeyGuard("secret").check("secret", "10.0.0.1") is True


def test_mauvaise_cle_refusee():
    assert ApiKeyGuard("secret").check("autre", "10.0.0.1") is False


def test_cle_absente_refusee():
    assert ApiKeyGuard("secret").check(None, "10.0.0.1") is False


def test_cle_vide_refusee():
    assert ApiKeyGuard("secret").check("", "10.0.0.1") is False


# ── Limitation des tentatives ─────────────────────────────────────────────────


def test_blocage_apres_le_quota_d_echecs():
    guard = ApiKeyGuard("secret", max_attempts=3)
    for _ in range(3):
        guard.check("faux", "10.0.0.1")
    assert guard.is_blocked("10.0.0.1") is True


def test_ip_bloquee_refusee_meme_avec_la_bonne_cle():
    guard = ApiKeyGuard("secret", max_attempts=2)
    guard.check("faux", "10.0.0.1")
    guard.check("faux", "10.0.0.1")
    assert guard.check("secret", "10.0.0.1") is False


def test_le_blocage_est_par_adresse():
    guard = ApiKeyGuard("secret", max_attempts=2)
    guard.check("faux", "10.0.0.1")
    guard.check("faux", "10.0.0.1")
    assert guard.is_blocked("10.0.0.2") is False
    assert guard.check("secret", "10.0.0.2") is True


def test_un_succes_remet_le_compteur_a_zero():
    guard = ApiKeyGuard("secret", max_attempts=3)
    guard.check("faux", "10.0.0.1")
    guard.check("secret", "10.0.0.1")
    guard.check("faux", "10.0.0.1")
    assert guard.is_blocked("10.0.0.1") is False


def test_le_blocage_expire_avec_la_fenetre():
    guard = ApiKeyGuard("secret", max_attempts=2, window_seconds=0.05)
    guard.check("faux", "10.0.0.1")
    guard.check("faux", "10.0.0.1")
    assert guard.is_blocked("10.0.0.1") is True

    import time

    time.sleep(0.06)
    assert guard.is_blocked("10.0.0.1") is False


def test_reset_debloque():
    guard = ApiKeyGuard("secret", max_attempts=1)
    guard.check("faux", "10.0.0.1")
    guard.reset("10.0.0.1")
    assert guard.is_blocked("10.0.0.1") is False
```

- [ ] **Step 2: Vérifier que les tests échouent**

Run: `.venv/bin/python -m pytest tests/api/test_auth.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'face_recognition_app.api'`

- [ ] **Step 3: Implémenter**

Créer `src/face_recognition_app/api/__init__.py` : fichier vide.

Créer `src/face_recognition_app/api/auth.py` :

```python
"""
auth.py
Authentification de l'API locale.

L'API exposait auparavant les flux caméra, l'historique nominatif et l'arrêt de
la surveillance sur `0.0.0.0:5000`, sans aucun contrôle. Toute requête exige
désormais un en-tête `X-API-Key`.

Origine de la clé, par ordre de priorité :
  1. la variable d'environnement `FR_API_KEY` ;
  2. le fichier `<config_dir>/api_key`, créé en 0600 au premier lancement.
"""

from __future__ import annotations

import hmac
import logging
import os
import secrets
import threading
import time

from ..settings import AppSettings

logger = logging.getLogger(__name__)

KEY_LENGTH = 32


def resolve_api_key(settings: AppSettings) -> str:
    """Clé de l'API : environnement, puis fichier, puis génération."""
    depuis_env = os.environ.get("FR_API_KEY", "").strip()
    if depuis_env:
        return depuis_env

    fichier = settings.api_key_file
    if fichier.exists():
        existante = fichier.read_text(encoding="utf-8").strip()
        if existante:
            return existante

    nouvelle = secrets.token_urlsafe(KEY_LENGTH)
    fichier.parent.mkdir(parents=True, exist_ok=True)
    fichier.write_text(nouvelle, encoding="utf-8")
    os.chmod(fichier, 0o600)
    logger.info("Clé d'API générée dans %s", fichier)
    return nouvelle


class ApiKeyGuard:
    """
    Vérifie la clé et limite les tentatives.

    La comparaison utilise `hmac.compare_digest` : une comparaison naïve fuite
    la longueur du préfixe correct par son temps d'exécution.
    """

    def __init__(
        self,
        api_key: str,
        max_attempts: int = 10,
        window_seconds: float = 300.0,
    ) -> None:
        self._key = api_key
        self._max_attempts = max_attempts
        self._window = window_seconds
        self._failures: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _recent_failures(self, client_ip: str, now: float) -> list[float]:
        """Échecs de cette adresse encore dans la fenêtre. Purge les plus anciens."""
        recents = [t for t in self._failures.get(client_ip, []) if now - t < self._window]
        if recents:
            self._failures[client_ip] = recents
        else:
            self._failures.pop(client_ip, None)
        return recents

    def is_blocked(self, client_ip: str) -> bool:
        with self._lock:
            return len(self._recent_failures(client_ip, time.monotonic())) >= self._max_attempts

    def reset(self, client_ip: str) -> None:
        with self._lock:
            self._failures.pop(client_ip, None)

    def check(self, provided: str | None, client_ip: str) -> bool:
        """Valide la clé fournie. Une adresse bloquée est refusée sans comparaison."""
        now = time.monotonic()
        with self._lock:
            if len(self._recent_failures(client_ip, now)) >= self._max_attempts:
                logger.warning("Requête refusée : %s a dépassé le quota de tentatives", client_ip)
                return False

            valide = bool(provided) and hmac.compare_digest(provided or "", self._key)
            if valide:
                self._failures.pop(client_ip, None)
                return True

            self._failures.setdefault(client_ip, []).append(now)
            logger.warning("Clé d'API invalide depuis %s", client_ip)
            return False
```

- [ ] **Step 4: Vérifier que les tests passent**

Run: `.venv/bin/python -m pytest tests/api/test_auth.py`
Expected: `17 passed`

- [ ] **Step 5: Commit**

```bash
git add src/face_recognition_app/api tests/api
git commit -m "feat: api/auth.py, cle d'API et limitation des tentatives

Cle lue depuis FR_API_KEY, sinon depuis <config_dir>/api_key genere en
0600 au premier lancement. Comparaison par hmac.compare_digest pour ne
pas fuiter la longueur du prefixe correct par le temps d'execution.

Blocage d'une adresse apres 10 echecs en 5 minutes ; un succes remet le
compteur a zero."
```

---

### Task 4: `api/server.py` — serveur arrêtable et routes protégées (bug ④)

**Files:**
- Create: `src/face_recognition_app/api/server.py`
- Create: `tests/api/test_server.py`
- Delete: `src/face_recognition_app/services/api_server.py`

**Interfaces:**
- Consumes: `AppSettings`, `ApiKeyGuard`, `resolve_api_key` (Task 3), `EventRepository` (Phase 1)
- Produces: `ApiServer(settings, camera_manager, engine, event_repository, recorder)` avec :
  - `start() -> bool` / `stop() -> None` (idempotents)
  - propriétés `is_running: bool`, `api_key: str`, `url: str`
  - `app` — l'application Flask, pour `app.test_client()`

**Deux défauts corrigés.** `stop()` ne faisait que journaliser : le thread daemon Flask continuait d'écouter, tandis qu'`is_running` restait `True`. Le bouton affichait `API: OFF` avec le serveur toujours actif, et un second clic ne redémarrait rien — l'API ne pouvait jamais être réactivée. `werkzeug.serving.make_server` expose un `shutdown()` réel.

Et l'écoute passe de `0.0.0.0` codé en dur à `settings.api_host`, qui vaut `127.0.0.1`.

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/api/test_server.py` :

```python
import numpy as np
import pytest

from face_recognition_app.api.server import ApiServer
from face_recognition_app.domain.camera import CameraConfig
from face_recognition_app.domain.detection import DetectedFace
from face_recognition_app.settings import AppSettings
from face_recognition_app.storage.event_repository import EventRepository


class FauxCameraManager:
    def __init__(self):
        self._configs = [CameraConfig(name="Salon", source_type="webcam", source=0, uid="cam1")]
        self.demarrages = 0
        self.arrets = 0

    def list_configs(self):
        return self._configs

    def get_config(self, uid):
        return next((c for c in self._configs if c.uid == uid), None)

    def is_running(self, uid):
        return uid == "cam1"

    def get_all_sources(self):
        return {"cam1": object()}

    def get_frame(self, uid):
        return np.zeros((48, 64, 3), dtype=np.uint8) if uid == "cam1" else None

    def start_all(self):
        self.demarrages += 1

    def stop_all(self):
        self.arrets += 1


class FauxMoteur:
    def __init__(self):
        self.actif = False

    @property
    def is_running(self):
        return self.actif

    def start(self):
        self.actif = True

    def stop(self):
        self.actif = False


class FauxRecorder:
    def list_clips(self):
        return []


@pytest.fixture
def contexte(tmp_path, monkeypatch):
    monkeypatch.setenv("FR_API_KEY", "cle-de-test")
    settings = AppSettings.create(project_root=tmp_path)
    settings.ensure_directories()
    events = EventRepository(settings)
    serveur = ApiServer(settings, FauxCameraManager(), FauxMoteur(), events, FauxRecorder())
    yield serveur, events, settings
    events.stop()


@pytest.fixture
def client(contexte):
    serveur, _, _ = contexte
    return serveur.app.test_client()


EN_TETES = {"X-API-Key": "cle-de-test"}


# ── Authentification ──────────────────────────────────────────────────────────


def test_sans_cle_refuse(client):
    assert client.get("/api/status").status_code == 401


def test_mauvaise_cle_refuse(client):
    assert client.get("/api/status", headers={"X-API-Key": "faux"}).status_code == 401


def test_bonne_cle_acceptee(client):
    assert client.get("/api/status", headers=EN_TETES).status_code == 200


def test_toutes_les_routes_de_lecture_sont_protegees(client):
    for route in ("/api/status", "/api/cameras", "/api/events", "/api/faces", "/api/clips"):
        assert client.get(route).status_code == 401, route


def test_blocage_apres_trop_d_echecs(client):
    for _ in range(11):
        client.get("/api/status", headers={"X-API-Key": "faux"})
    assert client.get("/api/status", headers=EN_TETES).status_code == 429


# ── Lecture ───────────────────────────────────────────────────────────────────


def test_status(client):
    data = client.get("/api/status", headers=EN_TETES).get_json()
    assert data["cameras_total"] == 1
    assert data["surveillance_active"] is False


def test_liste_des_cameras(client):
    data = client.get("/api/cameras", headers=EN_TETES).get_json()
    assert data[0]["name"] == "Salon"
    assert data[0]["running"] is True


def test_snapshot_retourne_du_jpeg(client):
    r = client.get("/api/snapshot/cam1", headers=EN_TETES)
    assert r.status_code == 200
    assert r.mimetype == "image/jpeg"


def test_snapshot_camera_inconnue(client):
    assert client.get("/api/snapshot/absente", headers=EN_TETES).status_code == 404


def test_evenements(contexte, client):
    _, events, _ = contexte
    events.record_sync(
        1000.0,
        "cam1",
        "Salon",
        [DetectedFace(location=(0, 1, 1, 0), name="Alice", confidence=0.9, is_known=True)],
    )
    data = client.get("/api/events", headers=EN_TETES).get_json()
    assert len(data) == 1
    assert data[0]["faces"][0]["name"] == "Alice"


def test_detail_d_un_evenement(contexte, client):
    _, events, _ = contexte
    evt = events.record_sync(1000.0, "cam1", "Salon", [])
    data = client.get(f"/api/events/{evt.id}", headers=EN_TETES).get_json()
    assert data["camera_name"] == "Salon"


def test_evenement_inconnu(client):
    assert client.get("/api/events/9999", headers=EN_TETES).status_code == 404


def test_limite_des_evenements_plafonnee(client):
    assert client.get("/api/events?limit=99999", headers=EN_TETES).status_code == 200


def test_limite_non_numerique_rejetee(client):
    assert client.get("/api/events?limit=beaucoup", headers=EN_TETES).status_code == 400


# ── Écriture, désactivée par défaut ───────────────────────────────────────────


def test_controle_refuse_par_defaut(client):
    assert client.post("/api/surveillance/start", headers=EN_TETES).status_code == 403


def test_controle_autorise_quand_active(tmp_path, monkeypatch):
    monkeypatch.setenv("FR_API_KEY", "cle-de-test")
    monkeypatch.setenv("FR_API_ALLOW_CONTROL", "true")
    settings = AppSettings.create(project_root=tmp_path)
    settings.ensure_directories()
    events = EventRepository(settings)
    moteur = FauxMoteur()
    try:
        serveur = ApiServer(settings, FauxCameraManager(), moteur, events, FauxRecorder())
        client = serveur.app.test_client()
        assert client.post("/api/surveillance/start", headers=EN_TETES).status_code == 200
        assert moteur.is_running is True
        assert client.post("/api/surveillance/stop", headers=EN_TETES).status_code == 200
        assert moteur.is_running is False
    finally:
        events.stop()


def test_controle_exige_aussi_la_cle(tmp_path, monkeypatch):
    monkeypatch.setenv("FR_API_KEY", "cle-de-test")
    monkeypatch.setenv("FR_API_ALLOW_CONTROL", "true")
    settings = AppSettings.create(project_root=tmp_path)
    settings.ensure_directories()
    events = EventRepository(settings)
    try:
        serveur = ApiServer(settings, FauxCameraManager(), FauxMoteur(), events, FauxRecorder())
        assert serveur.app.test_client().post("/api/surveillance/start").status_code == 401
    finally:
        events.stop()


# ── Cycle de vie ──────────────────────────────────────────────────────────────


def test_ecoute_locale_par_defaut(contexte):
    serveur, _, settings = contexte
    assert settings.api_host == "127.0.0.1"
    assert serveur.url.startswith("http://127.0.0.1:")


def test_demarrage_puis_arret_reel(contexte):
    serveur, _, _ = contexte
    assert serveur.start() is True
    assert serveur.is_running is True
    serveur.stop()
    assert serveur.is_running is False


def test_redemarrage_possible_apres_arret(contexte):
    """Le defaut historique : stop() ne fermait rien et is_running restait vrai."""
    serveur, _, _ = contexte
    serveur.start()
    serveur.stop()
    assert serveur.start() is True
    assert serveur.is_running is True
    serveur.stop()


def test_start_et_stop_idempotents(contexte):
    serveur, _, _ = contexte
    serveur.start()
    serveur.start()
    serveur.stop()
    serveur.stop()
    assert serveur.is_running is False
```

- [ ] **Step 2: Vérifier que les tests échouent**

Run: `.venv/bin/python -m pytest tests/api/test_server.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'face_recognition_app.api.server'`

- [ ] **Step 3: Implémenter**

Créer `src/face_recognition_app/api/server.py` :

```python
"""
server.py
API REST locale — fermée par défaut, ouverte explicitement.

Trois changements par rapport à `services/api_server.py`, qui est supprimé :

  - **Authentification.** Toute route exige `X-API-Key`. L'API exposait
    auparavant les flux caméra, l'historique nominatif et l'arrêt de la
    surveillance sans aucun contrôle.
  - **Écoute locale.** L'hôte vient de `settings.api_host`, à `127.0.0.1` par
    défaut, au lieu d'un `0.0.0.0` codé en dur.
  - **Arrêt réel.** `werkzeug.serving.make_server` expose `shutdown()`. La
    version précédente se contentait de journaliser : le thread continuait
    d'écouter, `is_running` restait vrai, et l'API ne pouvait plus être
    réactivée.

Les routes d'écriture sont gouvernées par `settings.api_allow_control`, à
`false` par défaut : consulter ses caméras à distance n'oblige pas à laisser
la télécommande ouverte.

Endpoints :
  GET  /api/status              état général
  GET  /api/cameras             caméras configurées
  GET  /api/snapshot/<uid>      dernière frame (JPEG)
  GET  /api/events              événements récents (limit, camera, person)
  GET  /api/events/<id>         détail + snapshot base64
  GET  /api/faces               personnes enregistrées
  GET  /api/clips               clips disponibles
  POST /api/surveillance/start  démarrer   (si api_allow_control)
  POST /api/surveillance/stop   arrêter    (si api_allow_control)
"""

from __future__ import annotations

import logging
import threading
from functools import wraps
from typing import TYPE_CHECKING, Any

import cv2

from ..settings import AppSettings
from .auth import ApiKeyGuard, resolve_api_key

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..services.camera_manager import CameraManager
    from ..services.video_recorder import VideoRecorder
    from ..storage.event_repository import EventRepository

MAX_EVENTS = 200
SNAPSHOT_QUALITY = 70


class ApiServer:
    """Serveur Flask embarqué, authentifié et réellement arrêtable."""

    VERSION = "2.0"

    def __init__(
        self,
        settings: AppSettings,
        camera_manager: CameraManager,
        engine: Any,
        event_repository: EventRepository,
        recorder: VideoRecorder,
    ) -> None:
        self._settings = settings
        self._mgr = camera_manager
        self._engine = engine
        self._events = event_repository
        self._recorder = recorder

        self._key = resolve_api_key(settings)
        self._guard = ApiKeyGuard(self._key)

        self._server: Any = None
        self._thread: threading.Thread | None = None

        self.app = self._build_app()

    # ── Propriétés ────────────────────────────────────────────────────────────

    @property
    def api_key(self) -> str:
        return self._key

    @property
    def url(self) -> str:
        return f"http://{self._settings.api_host}:{self._settings.api_port}"

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Construction ──────────────────────────────────────────────────────────

    def _build_app(self):
        from flask import Flask, Response, jsonify, request

        app = Flask(__name__)
        app.config["JSON_SORT_KEYS"] = False

        def authentifie(vue):
            """Exige une clé valide et applique la limitation par adresse."""

            @wraps(vue)
            def enveloppe(*args, **kwargs):
                ip = request.remote_addr or "inconnue"
                if self._guard.is_blocked(ip):
                    return jsonify({"error": "Trop de tentatives"}), 429
                if not self._guard.check(request.headers.get("X-API-Key"), ip):
                    return jsonify({"error": "Clé d'API invalide ou absente"}), 401
                return vue(*args, **kwargs)

            return enveloppe

        def controle_autorise(vue):
            """Refuse les routes d'écriture tant que api_allow_control est faux."""

            @wraps(vue)
            def enveloppe(*args, **kwargs):
                if not self._settings.api_allow_control:
                    return jsonify(
                        {"error": "Contrôle à distance désactivé (FR_API_ALLOW_CONTROL)"}
                    ), 403
                return vue(*args, **kwargs)

            return enveloppe

        # ── Statut ────────────────────────────────────────────────────────────

        @app.route("/api/status")
        @authentifie
        def status():
            return jsonify(
                {
                    "version": self.VERSION,
                    "surveillance_active": bool(getattr(self._engine, "is_running", False)),
                    "cameras_total": len(self._mgr.list_configs()),
                    "cameras_running": len(self._mgr.get_all_sources()),
                    "control_enabled": self._settings.api_allow_control,
                }
            )

        # ── Caméras ───────────────────────────────────────────────────────────

        @app.route("/api/cameras")
        @authentifie
        def cameras():
            return jsonify(
                [
                    {**c.to_dict(), "running": self._mgr.is_running(c.uid)}
                    for c in self._mgr.list_configs()
                ]
            )

        @app.route("/api/snapshot/<uid>")
        @authentifie
        def snapshot(uid: str):
            frame = self._mgr.get_frame(uid)
            if frame is None:
                return jsonify({"error": "Caméra indisponible"}), 404
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, SNAPSHOT_QUALITY])
            if not ok:
                logger.error("Encodage JPEG du snapshot de %s échoué", uid)
                return jsonify({"error": "Encodage échoué"}), 500
            return Response(buf.tobytes(), mimetype="image/jpeg")

        # ── Événements ────────────────────────────────────────────────────────

        @app.route("/api/events")
        @authentifie
        def events():
            try:
                limit = min(int(request.args.get("limit", 50)), MAX_EVENTS)
            except ValueError:
                return jsonify({"error": "Le paramètre 'limit' doit être un entier"}), 400

            personne = request.args.get("person")
            camera = request.args.get("camera")
            if personne:
                trouves = self._events.get_by_person(personne, limit)
            elif camera:
                trouves = self._events.get_by_camera(camera, limit)
            else:
                trouves = self._events.get_recent(limit)

            return jsonify(
                [
                    {
                        "id": e.id,
                        "datetime": e.dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "camera_uid": e.camera_uid,
                        "camera_name": e.camera_name,
                        "faces": [f.to_dict() for f in e.faces],
                        "has_snapshot": e.snapshot_b64 is not None,
                    }
                    for e in trouves
                ]
            )

        @app.route("/api/events/<int:event_id>")
        @authentifie
        def event_detail(event_id: int):
            evt = self._events.get_by_id(event_id)
            if evt is None:
                return jsonify({"error": "Événement introuvable"}), 404
            return jsonify(evt.to_dict())

        # ── Personnes et clips ────────────────────────────────────────────────

        @app.route("/api/faces")
        @authentifie
        def faces():
            from ..storage.encodings_repository import EncodingsRepository

            metadata = EncodingsRepository(self._settings).load_metadata()
            return jsonify([{"uid": uid, **info} for uid, info in metadata.items()])

        @app.route("/api/clips")
        @authentifie
        def clips():
            return jsonify(
                [
                    {"name": p.name, "size_mb": round(p.stat().st_size / 1_048_576, 2)}
                    for p in self._recorder.list_clips()
                ]
            )

        # ── Contrôle de la surveillance ───────────────────────────────────────

        @app.route("/api/surveillance/start", methods=["POST"])
        @authentifie
        @controle_autorise
        def surv_start():
            if not getattr(self._engine, "is_running", False):
                self._mgr.start_all()
                self._engine.start()
            return jsonify({"ok": True, "active": True})

        @app.route("/api/surveillance/stop", methods=["POST"])
        @authentifie
        @controle_autorise
        def surv_stop():
            if getattr(self._engine, "is_running", False):
                self._engine.stop()
                self._mgr.stop_all()
            return jsonify({"ok": True, "active": False})

        return app

    # ── Cycle de vie ──────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Lance le serveur. Idempotent. Retourne True si le serveur écoute."""
        if self.is_running:
            return True

        from werkzeug.serving import make_server

        hote, port = self._settings.api_host, self._settings.api_port
        try:
            self._server = make_server(hote, port, self.app, threaded=True)
        except OSError as exc:
            logger.error("Impossible d'écouter sur %s:%d : %s", hote, port, exc)
            self._server = None
            return False

        self._thread = threading.Thread(
            target=self._server.serve_forever, name="api-server", daemon=True
        )
        self._thread.start()

        if hote not in ("127.0.0.1", "localhost"):
            logger.warning(
                "API exposée sur %s — accessible depuis le réseau. "
                "La clé d'API est le seul rempart.",
                hote,
            )
        logger.info("API REST démarrée sur %s/api/status", self.url)
        return True

    def stop(self) -> None:
        """Ferme réellement l'écoute. Idempotent."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("API REST arrêtée")
```

- [ ] **Step 4: Vérifier que les tests passent**

Run: `.venv/bin/python -m pytest tests/api/test_server.py`
Expected: `22 passed`

- [ ] **Step 5: Supprimer l'ancien module**

```bash
git rm src/face_recognition_app/services/api_server.py
```

- [ ] **Step 6: Commit**

```bash
git add -A src/face_recognition_app tests/api
git commit -m "feat: api/server.py authentifie et reellement arretable (bug 4)

L'API exposait les flux camera, l'historique nominatif et l'arret de la
surveillance sur 0.0.0.0:5000 sans aucun controle. Toute route exige
desormais X-API-Key, et l'ecoute se fait sur settings.api_host, soit
127.0.0.1 par defaut.

stop() ne faisait que journaliser : le thread daemon Flask continuait
d'ecouter tandis qu'is_running restait vrai, si bien que l'API ne pouvait
jamais etre reactivee. make_server expose un shutdown() reel.

Les routes d'ecriture sont soumises a api_allow_control, faux par
defaut : consulter ses cameras a distance n'oblige plus a laisser la
telecommande ouverte."
```

---

### Task 5: Câblage du dashboard et du point d'entrée

**Files:**
- Modify: `src/face_recognition_app/ui/surveillance_dashboard.py`
- Modify: `src/face_recognition_app/__main__.py`

**Interfaces:**
- Consumes: `ApiServer` (Task 4), `AppContext` (Phase 1)
- Produces: rien de nouveau — le dashboard consomme le nouveau serveur.

- [ ] **Step 1: Remplacer l'import de l'ancien serveur**

Dans `src/face_recognition_app/ui/surveillance_dashboard.py`, remplacer :

```python
from ..services.api_server import ApiServer
```

par :

```python
from ..api.server import ApiServer
```

- [ ] **Step 2: Adapter la construction**

Toujours dans le même fichier, remplacer la ligne de construction :

```python
        self._api_server = ApiServer(self._cam_mgr, self._engine, self._event_store, self._recorder)
```

par :

```python
        self._api_server = ApiServer(
            AppSettings.create(),
            self._cam_mgr,
            self._engine,
            self._event_store.repository,
            self._recorder,
        )
```

et ajouter en tête du fichier, avec les autres imports relatifs :

```python
from ..settings import AppSettings
```

- [ ] **Step 3: Réécrire la bascule de l'API**

Remplacer la méthode `_toggle_api` par :

```python
    def _toggle_api(self) -> None:
        if self._api_server.is_running:
            self._api_server.stop()
            self._api_var.set("API: OFF")
            self._api_btn.configure(bg="#555")
            return

        if not self._api_server.start():
            messagebox.showerror(
                "API",
                f"Impossible d'écouter sur {self._api_server.url}.\n"
                "Le port est peut-être déjà utilisé.",
                parent=self,
            )
            return

        self._api_var.set(f"API: {self._api_server.url.rsplit(':', 1)[1]}")
        self._api_btn.configure(bg="#27ae60")
        self._afficher_cle_api()

    def _afficher_cle_api(self) -> None:
        """Montre l'URL et la clé, avec un bouton de copie."""
        fenetre = tk.Toplevel(self)
        fenetre.title("Accès à l'API")
        fenetre.resizable(False, False)
        fenetre.transient(self)

        tk.Label(
            fenetre,
            text="Envoyez cette clé dans l'en-tête X-API-Key :",
            font=("Helvetica", 10),
        ).pack(padx=16, pady=(14, 6))

        cle = self._api_server.api_key
        champ = ttk.Entry(fenetre, width=52, justify=tk.CENTER)
        champ.insert(0, cle)
        champ.configure(state="readonly")
        champ.pack(padx=16)

        tk.Label(
            fenetre,
            text=f"{self._api_server.url}/api/status",
            fg="#555",
            font=("Helvetica", 9),
        ).pack(pady=(6, 0))

        def copier() -> None:
            self.clipboard_clear()
            self.clipboard_append(cle)

        barre = tk.Frame(fenetre)
        barre.pack(pady=12)
        ttk.Button(barre, text="Copier la clé", command=copier).pack(side=tk.LEFT, padx=4)
        ttk.Button(barre, text="Fermer", command=fenetre.destroy).pack(side=tk.LEFT, padx=4)
```

- [ ] **Step 4: Exposer `is_running` sur le moteur**

`ApiServer` interroge `engine.is_running`, mais `SurveillanceEngine` n'expose que
l'attribut privé `_running` — que l'ancien `api_server.py` et le dashboard lisaient
directement. Sans propriété publique, `/api/status` rapporterait toujours `false`.

Dans `src/face_recognition_app/services/surveillance_engine.py`, ajouter juste après
la méthode `set_alert_manager` :

```python
    @property
    def is_running(self) -> bool:
        """Le moteur analyse-t-il actuellement ?"""
        return self._running
```

Puis remplacer les deux accès privés restants dans
`src/face_recognition_app/ui/surveillance_dashboard.py` — `if self._engine._running:`
devient `if self._engine.is_running:` (deux occurrences, dans `_add_camera` et
`_refresh_tiles`).

Run: `grep -rn "_engine._running" src/`
Expected: aucune ligne.

- [ ] **Step 5: Fermer l'API à la fermeture du dashboard**

Vérifier que `_on_close` contient bien `self._api_server.stop()`. C'est déjà le cas ; l'appel est maintenant effectif au lieu d'être une simple journalisation.

- [ ] **Step 6: Vérifier l'ensemble**

Run: `.venv/bin/ruff check src tests && .venv/bin/ruff format src tests && .venv/bin/mypy && .venv/bin/python -m pytest`
Expected: `All checks passed!`, `Success: no issues found`, `170 passed`

- [ ] **Step 7: Vérifier que l'application démarre**

Run: `timeout 12 .venv/bin/python main.py; echo "code: $?"`
Expected: `code: 124` — la fenêtre s'est ouverte et le timeout l'a interrompue. Tout autre code signale une exception à corriger.

- [ ] **Step 8: Vérifier l'API de bout en bout, sur une copie temporaire**

**Ne pas exécuter contre la racine du projet.**

```bash
BANC=$(mktemp -d)
FR_API_KEY=essai .venv/bin/python - "$BANC" <<'PY'
import sys, pathlib, time, urllib.request, urllib.error
sys.path.insert(0, "src")
banc = pathlib.Path(sys.argv[1])

from face_recognition_app.settings import AppSettings
from face_recognition_app.storage.event_repository import EventRepository
from face_recognition_app.api.server import ApiServer

class FauxMgr:
    def list_configs(self): return []
    def get_all_sources(self): return {}
    def is_running(self, uid): return False
    def get_frame(self, uid): return None
    def start_all(self): pass
    def stop_all(self): pass

class FauxMoteur:
    is_running = False

class FauxRec:
    def list_clips(self): return []

s = AppSettings.create(project_root=banc)
s.ensure_directories()
events = EventRepository(s)
srv = ApiServer(s, FauxMgr(), FauxMoteur(), events, FauxRec())
assert srv.start()
time.sleep(0.3)

def appel(cle):
    req = urllib.request.Request(f"{srv.url}/api/status")
    if cle:
        req.add_header("X-API-Key", cle)
    try:
        with urllib.request.urlopen(req, timeout=3) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code

print("sans cle        :", appel(None))
print("bonne cle       :", appel("essai"))
srv.stop()
print("arrete          :", not srv.is_running)
assert srv.start(), "redemarrage impossible"
print("redemarrage     : ok")
srv.stop()
events.stop()
PY
rm -rf "$BANC"
```

Expected :
```
sans cle        : 401
bonne cle       : 200
arrete          : True
redemarrage     : ok
```

- [ ] **Step 9: Commit**

```bash
git add -A src/face_recognition_app/ui src/face_recognition_app/__main__.py
git commit -m "refactor: dashboard cable sur l'API authentifiee

Le bouton API affiche desormais la cle avec un bouton de copie, signale
un port deja occupe au lieu d'echouer en silence, et l'arret ferme
reellement l'ecoute."
```

---

## Vérification de fin de Phase 2A

- [ ] **Suite complète et qualité**

Run: `.venv/bin/python -m pytest && .venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests && .venv/bin/mypy`
Expected: `170 passed`, `All checks passed!`, `Success: no issues found`

- [ ] **Aucun test n'ouvre de caméra ni ne lie de port**

Run: `.venv/bin/python -m pytest --durations=5`
Expected: aucune durée supérieure à 3 s.

- [ ] **Données réelles intactes**

Run: `.venv/bin/python -c "
import sys; sys.path.insert(0,'src')
from face_recognition_app.settings import AppSettings
from face_recognition_app.storage.encodings_repository import EncodingsRepository
from face_recognition_app.storage.camera_repository import CameraRepository
s = AppSettings.create()
assert sorted(EncodingsRepository(s).load_map()) == ['ravaka', 'vic', 'victorien']
assert len(CameraRepository(s).load_all()) == 4
print('donnees intactes')
" 2>&1 | grep -v '^2026-'`
Expected: `donnees intactes`

- [ ] **L'application démarre**

Run: `timeout 12 .venv/bin/python main.py; echo "code: $?"`
Expected: `code: 124`

---

## Ce que la Phase 2A ne fait pas

- Le moteur applique toujours partiellement le profil : `record_video`, les pré/post-enregistrements et les filtres d'alerte restent sans effet (bugs ⑤ ⑥). Phase 2B.
- Les callbacks `on_change` s'accumulent toujours à chaque redémarrage du moteur (bug ⑦). Phase 2B.
- Les écouteurs du moteur sont toujours appelés en synchrone depuis le thread d'analyse : le bus d'événements arrive en Phase 2B.
- Les clips restent enregistrés à 15 fps pour un producteur à 30 fps, soit une lecture à vitesse double. Phase 2B.
- L'alarme sonore n'est pas encore un canal d'`AlertManager` (bug ⑧). Phase 2B.
- Les bugs ① ② ⑨ ⑩ ⑪ vivent dans la couche UI, traitée en Phase 3.
- La vignette n'affiche pas encore « reconnexion dans N s » : `CameraSource.next_retry_in`
  est exposé, mais son câblage dans `CameraTile` relève de la Phase 3.
- Le mot de passe SMTP reste en clair dans `alerts_config.json` (spec §6.2) : `AlertManager`
  est refondu en Phase 2B.
- Les adaptateurs `storage/config.py`, `encodings_store.py`, `event_store.py`, `profile_store.py` subsistent jusqu'à la Phase 3.
