# Refactorisation — Phase 1 : Socle et stockage — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Poser le socle testable de l'application — outillage, configuration injectée, domaine pur sans dépendance matérielle, et repositories de stockage robustes — sans changer le comportement visible de l'application.

**Architecture:** On construit les nouvelles couches `settings.py`, `domain/` et `storage/*_repository.py` à côté de l'existant. Les anciens modules (`storage/config.py`, `storage/encodings_store.py`, `storage/event_store.py`, `storage/profile_store.py`, `core/utils.py`) deviennent des adaptateurs minces qui délèguent aux nouveaux, afin que `services/` et `ui/` continuent de fonctionner sans modification. Ces adaptateurs seront supprimés en Phase 2 et 3.

**Tech Stack:** Python 3.12, numpy, SQLite (stdlib), pytest, ruff. Le domaine n'importe ni dlib, ni OpenCV, ni Tkinter.

## Global Constraints

- Python `>=3.10` (le code utilise `str | int`, syntaxe 3.10+). Corriger `requires-python` dans `pyproject.toml`, qui déclare `>=3.8` à tort.
- `domain/` n'importe **jamais** `face_recognition`, `dlib`, `cv2` ni `tkinter`. Seuls `numpy` et la stdlib sont autorisés.
- Aucun module ne lit de configuration à l'import. Toute configuration passe par un objet `AppSettings` injecté.
- Aucun `except Exception: pass` nu. Toute exception est journalisée avec son contexte.
- Les données existantes restent lisibles sans intervention : `encodings/*.json` (4 fichiers), `events.db` (8,5 Mo), `cameras.json`, `profiles.json`.
- Tout commit laisse `pytest -q` vert et `python main.py` fonctionnel.
- Messages de commit en français, préfixés `feat:`, `fix:`, `refactor:`, `test:` ou `chore:`.
- Deux seuils seulement dans tout le projet : `settings.duplicate_tolerance` (enrôlement) et `profile.recognition_threshold` (reconnaissance live). Les constantes `FACE_RECOGNITION_THRESHOLD`, `DUPLICATE_TOLERANCE` et `VIDEO_FACE_TOLERANCE` de `storage/config.py` sont remplacées par ces deux-là.

---

### Task 1: Outillage — pytest, dépendances épinglées, ruff

**Files:**
- Modify: `pyproject.toml`
- Modify: `requirements.txt`
- Create: `requirements-dev.txt`
- Delete: `tests/conftest.py`

**Interfaces:**
- Consumes: rien
- Produces: `pytest` exécutable sans bricolage de `sys.path` ; `ruff` configuré.

- [ ] **Step 1: Vérifier l'état de départ**

Run: `.venv/bin/python -m pytest -q`
Expected: `14 passed`

- [ ] **Step 2: Configurer pytest et ruff dans `pyproject.toml`**

Remplacer le bloc `[project]` `requires-python` et ajouter les sections outillage. Le fichier complet devient :

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "face_recognition_app"
version = "0.2.0"
description = "Local face recognition surveillance application with Tkinter UI"
readme = "README.md"
requires-python = ">=3.10"
license = "MIT"
keywords = ["face recognition", "surveillance", "opencv", "tkinter", "security"]

dependencies = [
    "face_recognition",
    "opencv-python",
    "numpy",
    "Pillow",
    "pygame",
    "ttkbootstrap",
    "flask",
    "plyer",
    "requests",
]

[project.scripts]
face-recognition = "face_recognition_app.__main__:main"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
line-length = 100
target-version = "py310"
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "SIM"]
ignore = ["E501"]

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["B011"]

[tool.mypy]
python_version = "3.10"
files = ["src/face_recognition_app/domain", "src/face_recognition_app/settings.py"]
ignore_missing_imports = true
```

Note : `playsound` et `psutil` sont retirés de `dependencies` — jamais importés dans le code (`playsound` n'est même pas installé).

- [ ] **Step 3: Épingler `requirements.txt`**

```
face_recognition==1.3.0
opencv-python==4.13.0.92
numpy==2.4.4
Pillow==12.2.0
pygame==2.6.1
ttkbootstrap==1.20.2
flask==3.1.3
plyer==2.1.0
requests==2.33.1
```

- [ ] **Step 4: Créer `requirements-dev.txt`**

```
-r requirements.txt
pytest==9.0.2
ruff==0.9.6
mypy==1.15.0
```

- [ ] **Step 5: Supprimer le bricolage de `sys.path`**

`tests/conftest.py` ne contient que la manipulation de `sys.path`, désormais assurée par `pythonpath = ["src"]`.

```bash
rm tests/conftest.py
```

- [ ] **Step 6: Vérifier que les tests passent toujours**

Run: `.venv/bin/python -m pytest`
Expected: `14 passed`

- [ ] **Step 7: Installer ruff et formater**

```bash
.venv/bin/pip install ruff==0.9.6
.venv/bin/ruff check --fix src tests
.venv/bin/ruff format src tests
```

Run: `.venv/bin/ruff check src tests`
Expected: `All checks passed!`

- [ ] **Step 8: Revérifier les tests après formatage**

Run: `.venv/bin/python -m pytest`
Expected: `14 passed`

- [ ] **Step 9: Résoudre le conflit OpenCV (bug ⑫)**

`opencv-python` et `opencv-python-headless` sont installés simultanément dans le venv.
La variante headless l'emporte à l'import et casse silencieusement `cv2.imshow`.

Run: `.venv/bin/pip uninstall -y opencv-python-headless`
Expected: `Successfully uninstalled opencv-python-headless-4.13.0.92`

Vérifier que l'affichage OpenCV fonctionne à nouveau :

Run: `.venv/bin/python -c "import cv2; cv2.namedWindow('t'); cv2.destroyAllWindows(); print('backend GUI OK')"`
Expected: `backend GUI OK`

Si la commande échoue avec « The function is not implemented », réinstaller la variante
complète : `.venv/bin/pip install --force-reinstall opencv-python==4.13.0.92`.

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml requirements.txt requirements-dev.txt src tests
git rm --cached tests/conftest.py 2>/dev/null || true
git add -A
git commit -m "chore: outillage pytest et ruff, dependances epinglees

- pythonpath pytest remplace le bricolage sys.path de conftest.py
- versions epinglees, requirements-dev.txt separe
- playsound et psutil retires (jamais importes)
- requires-python corrige en >=3.10 (le code utilise la syntaxe str | int)"
```

---

### Task 2: `domain/matching.py` — comparaison de visages en numpy pur

**Files:**
- Create: `src/face_recognition_app/domain/__init__.py`
- Create: `src/face_recognition_app/domain/matching.py`
- Create: `tests/domain/test_matching.py`

**Interfaces:**
- Consumes: rien
- Produces:
  - `face_distance(known: np.ndarray, unknown: np.ndarray) -> np.ndarray`
  - `best_match(known: np.ndarray, unknown: np.ndarray, threshold: float) -> tuple[int | None, float]`
  - `find_duplicate(unknown: np.ndarray, existing: Sequence[tuple[str, np.ndarray]], tolerance: float) -> str | None`

**Pourquoi :** `core/utils.is_duplicate` importe `face_recognition`, donc dlib, donc ~10 min de compilation en CI. Or `compare_faces` et `face_distance` ne sont qu'une norme euclidienne et un seuil. Les réimplémenter en numpy retire dlib du domaine et rend la CI triviale.

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/domain/test_matching.py` :

```python
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
```

- [ ] **Step 2: Vérifier que les tests échouent**

Run: `.venv/bin/python -m pytest tests/domain/test_matching.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'face_recognition_app.domain'`

- [ ] **Step 3: Créer le paquet et l'implémentation**

`src/face_recognition_app/domain/__init__.py` : fichier vide.

`src/face_recognition_app/domain/matching.py` :

```python
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
```

- [ ] **Step 4: Vérifier que les tests passent**

Run: `.venv/bin/python -m pytest tests/domain/test_matching.py`
Expected: `11 passed`

- [ ] **Step 5: Vérifier l'absence de dépendance lourde**

Run: `.venv/bin/python -c "import sys; sys.path.insert(0,'src'); import face_recognition_app.domain.matching; assert 'dlib' not in sys.modules and 'cv2' not in sys.modules; print('domaine propre')"`
Expected: `domaine propre`

- [ ] **Step 6: Commit**

```bash
git add src/face_recognition_app/domain tests/domain
git commit -m "feat: domain/matching.py, comparaison de visages en numpy pur

Reimplemente face_distance et la detection de doublon sans dlib :
la distance entre encodages n'est qu'une norme euclidienne.

find_duplicate retourne desormais la correspondance la plus proche
et non la premiere rencontree."
```

---

### Task 3: `settings.py` — configuration injectée

**Files:**
- Create: `src/face_recognition_app/settings.py`
- Create: `tests/test_settings.py`

**Interfaces:**
- Consumes: rien
- Produces: `AppSettings` (dataclass gelée) avec `AppSettings.create(project_root: Path | None = None) -> AppSettings` et la méthode `ensure_directories() -> None`.

Champs produits, utilisés par toutes les tâches suivantes :
`project_root`, `config_dir`, `encodings_dir`, `events_db`, `clips_dir`, `cameras_file`, `profiles_file`, `alerts_file`, `api_key_file`, `capture_fps`, `ui_refresh_ms`, `event_retention_days`, `max_clips`, `max_clips_mb`, `api_host`, `api_port`, `api_allow_control`, `duplicate_tolerance`.

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/test_settings.py` :

```python
from pathlib import Path

from face_recognition_app.settings import AppSettings


def test_chemins_derives_de_la_racine(tmp_path):
    s = AppSettings.create(project_root=tmp_path)
    assert s.encodings_dir == tmp_path / "encodings"
    assert s.events_db == tmp_path / "events.db"
    assert s.clips_dir == tmp_path / "clips"
    assert s.cameras_file == tmp_path / "cameras.json"
    assert s.profiles_file == tmp_path / "profiles.json"


def test_valeurs_par_defaut_sures(tmp_path):
    s = AppSettings.create(project_root=tmp_path)
    assert s.api_host == "127.0.0.1"
    assert s.api_allow_control is False
    assert s.event_retention_days == 30
    assert s.capture_fps == 15.0


def test_surcharge_par_variables_d_environnement(tmp_path, monkeypatch):
    monkeypatch.setenv("FR_API_PORT", "8123")
    monkeypatch.setenv("FR_API_HOST", "0.0.0.0")
    monkeypatch.setenv("FR_EVENT_RETENTION_DAYS", "7")
    s = AppSettings.create(project_root=tmp_path)
    assert s.api_port == 8123
    assert s.api_host == "0.0.0.0"
    assert s.event_retention_days == 7


def test_variable_invalide_conserve_le_defaut(tmp_path, monkeypatch):
    monkeypatch.setenv("FR_API_PORT", "pas-un-nombre")
    s = AppSettings.create(project_root=tmp_path)
    assert s.api_port == 5000


def test_ensure_directories_cree_les_dossiers(tmp_path):
    s = AppSettings.create(project_root=tmp_path)
    s.ensure_directories()
    assert s.encodings_dir.is_dir()
    assert s.clips_dir.is_dir()
    assert s.config_dir.is_dir()


def test_settings_est_immuable(tmp_path):
    import dataclasses

    import pytest

    s = AppSettings.create(project_root=tmp_path)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.api_port = 1234
```

- [ ] **Step 2: Vérifier que les tests échouent**

Run: `.venv/bin/python -m pytest tests/test_settings.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'face_recognition_app.settings'`

- [ ] **Step 3: Implémenter**

`src/face_recognition_app/settings.py` :

```python
"""
settings.py
Configuration de l'application, construite une fois et injectée.

Aucun module ne doit lire la configuration à l'import : `AppSettings` est
construit dans `__main__` puis passé explicitement aux composants qui en ont
besoin. C'est ce qui rend le stockage testable sans monkey-patching de
constantes de module.

Surcharges par variables d'environnement, toutes préfixées `FR_` :
  FR_API_HOST, FR_API_PORT, FR_API_ALLOW_CONTROL,
  FR_EVENT_RETENTION_DAYS, FR_CAPTURE_FPS, FR_MAX_CLIPS
Les secrets ne transitent jamais par un fichier : FR_API_KEY, FR_SMTP_PASSWORD.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("%s='%s' n'est pas un entier, valeur par défaut %d conservée", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("%s='%s' n'est pas un nombre, valeur par défaut %s conservée", name, raw, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "oui"}


@dataclass(frozen=True)
class AppSettings:
    """Configuration complète de l'application. Immuable."""

    project_root: Path
    config_dir: Path
    encodings_dir: Path
    events_db: Path
    clips_dir: Path
    cameras_file: Path
    profiles_file: Path
    alerts_file: Path
    api_key_file: Path

    # Capture et affichage
    capture_fps: float = 15.0
    ui_refresh_ms: int = 200

    # Rétention
    event_retention_days: int = 30
    max_clips: int = 100
    max_clips_mb: int = 2048

    # API — fermée par défaut
    api_host: str = "127.0.0.1"
    api_port: int = 5000
    api_allow_control: bool = False

    # Seuil d'enrôlement. Le seuil de reconnaissance live vit dans le profil.
    duplicate_tolerance: float = 0.6

    @classmethod
    def create(cls, project_root: Path | None = None) -> AppSettings:
        """Construit la configuration à partir de la racine projet et de l'environnement."""
        root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
        config_dir = root / ".config"
        return cls(
            project_root=root,
            config_dir=config_dir,
            encodings_dir=root / "encodings",
            events_db=root / "events.db",
            clips_dir=root / "clips",
            cameras_file=root / "cameras.json",
            profiles_file=root / "profiles.json",
            alerts_file=root / "alerts_config.json",
            api_key_file=config_dir / "api_key",
            capture_fps=_env_float("FR_CAPTURE_FPS", 15.0),
            ui_refresh_ms=_env_int("FR_UI_REFRESH_MS", 200),
            event_retention_days=_env_int("FR_EVENT_RETENTION_DAYS", 30),
            max_clips=_env_int("FR_MAX_CLIPS", 100),
            max_clips_mb=_env_int("FR_MAX_CLIPS_MB", 2048),
            api_host=os.environ.get("FR_API_HOST", "127.0.0.1"),
            api_port=_env_int("FR_API_PORT", 5000),
            api_allow_control=_env_bool("FR_API_ALLOW_CONTROL", False),
            duplicate_tolerance=_env_float("FR_DUPLICATE_TOLERANCE", 0.6),
        )

    def ensure_directories(self) -> None:
        """Crée les répertoires de données. Le répertoire de config est en 0700."""
        self.encodings_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.chmod(0o700)
```

- [ ] **Step 4: Vérifier que les tests passent**

Run: `.venv/bin/python -m pytest tests/test_settings.py`
Expected: `6 passed`

- [ ] **Step 5: Ignorer le répertoire de configuration**

Ajouter à `.gitignore`, sous la section « Données générées par l'app » :

```
.config/
```

- [ ] **Step 6: Commit**

```bash
git add src/face_recognition_app/settings.py tests/test_settings.py .gitignore
git commit -m "feat: AppSettings, configuration injectee et immuable

Remplace les constantes de module de storage/config.py, que les tests
devaient muter. Valeurs par defaut sures : API sur 127.0.0.1, controle
a distance desactive, retention 30 jours."
```

---

### Task 4: `domain/camera.py` — configuration de caméra

**Files:**
- Create: `src/face_recognition_app/domain/camera.py`
- Create: `tests/domain/test_camera.py`

**Interfaces:**
- Consumes: rien
- Produces: `CameraConfig` avec `to_dict() -> dict`, `from_dict(data: dict) -> CameraConfig`, et la propriété `is_ip: bool`. Champs : `name`, `source_type`, `source`, `enabled`, `uid`, `width`, `height`, `roi`, `detection_model`.

Déplacé depuis `services/camera_source.py:30-76` sans changement de format sérialisé — `cameras.json` reste lisible tel quel.

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/domain/test_camera.py` :

```python
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
```

- [ ] **Step 2: Vérifier que les tests échouent**

Run: `.venv/bin/python -m pytest tests/domain/test_camera.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'face_recognition_app.domain.camera'`

- [ ] **Step 3: Implémenter**

`src/face_recognition_app/domain/camera.py` :

```python
"""
camera.py
Configuration persistable d'une caméra.

Le format sérialisé est identique à celui de `cameras.json` existant :
la refactorisation ne demande aucune migration de ce fichier.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

SOURCE_TYPES = frozenset({"webcam", "ip"})
DETECTION_MODELS = frozenset({"hog", "cnn"})


@dataclass
class CameraConfig:
    """Paramètres d'une caméra. `source` est un index entier (webcam) ou une URL (IP)."""

    name: str
    source_type: str
    source: str | int
    enabled: bool = True
    uid: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    width: int = 640
    height: int = 480

    # Zone d'intérêt (x, y, largeur, hauteur) en pixels ; None = toute l'image
    roi: tuple[int, int, int, int] | None = None

    detection_model: str = "hog"

    def __post_init__(self) -> None:
        if self.source_type not in SOURCE_TYPES:
            raise ValueError(
                f"source_type invalide : {self.source_type!r} (attendu : {sorted(SOURCE_TYPES)})"
            )
        if self.detection_model not in DETECTION_MODELS:
            raise ValueError(
                f"detection_model invalide : {self.detection_model!r} "
                f"(attendu : {sorted(DETECTION_MODELS)})"
            )
        if self.roi is not None:
            self.roi = tuple(int(v) for v in self.roi)  # type: ignore[assignment]

    @property
    def is_ip(self) -> bool:
        return self.source_type == "ip"

    def to_dict(self) -> dict:
        return {
            "uid": self.uid,
            "name": self.name,
            "source_type": self.source_type,
            "source": self.source,
            "enabled": self.enabled,
            "width": self.width,
            "height": self.height,
            "roi": list(self.roi) if self.roi else None,
            "detection_model": self.detection_model,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CameraConfig:
        roi_raw = data.get("roi")
        return cls(
            uid=data.get("uid", uuid.uuid4().hex[:8]),
            name=data["name"],
            source_type=data["source_type"],
            source=data["source"],
            enabled=data.get("enabled", True),
            width=data.get("width", 640),
            height=data.get("height", 480),
            roi=tuple(roi_raw) if roi_raw else None,
            detection_model=data.get("detection_model", "hog"),
        )
```

- [ ] **Step 4: Vérifier que les tests passent**

Run: `.venv/bin/python -m pytest tests/domain/test_camera.py`
Expected: `8 passed`

- [ ] **Step 5: Vérifier que le vrai `cameras.json` se relit**

Run: `.venv/bin/python -c "
import json, sys; sys.path.insert(0,'src')
from face_recognition_app.domain.camera import CameraConfig
data = json.loads(open('cameras.json').read())
for d in data:
    c = CameraConfig.from_dict(d)
    assert c.to_dict() == d, c.name
print(f'{len(data)} cameras relues sans perte')
"`
Expected: `4 cameras relues sans perte`

- [ ] **Step 6: Commit**

```bash
git add src/face_recognition_app/domain/camera.py tests/domain/test_camera.py
git commit -m "feat: domain/camera.py, CameraConfig avec validation

Deplace depuis services/camera_source.py. Format serialise inchange,
cameras.json reste lisible. Ajoute la validation de source_type et
detection_model, absente jusqu'ici."
```

---

### Task 5: `domain/detection.py` — visage détecté et événement

**Files:**
- Create: `src/face_recognition_app/domain/detection.py`
- Create: `tests/domain/test_detection.py`

**Interfaces:**
- Consumes: rien
- Produces:
  - `DetectedFace(location: tuple[int,int,int,int], name: str, confidence: float, is_known: bool)` avec `to_dict()` / `from_dict()`
  - `SurveillanceEvent(camera_uid, camera_name, timestamp, faces, motion_score=0.0, frame=None)` avec les propriétés `known_names`, `unknown_count`, `has_unknown`, `dt`
  - Constante `UNKNOWN_NAME = "Inconnu"`

`frame` est typé `Any` afin que le domaine n'importe pas numpy pour un simple transport d'image.

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/domain/test_detection.py` :

```python
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
```

- [ ] **Step 2: Vérifier que les tests échouent**

Run: `.venv/bin/python -m pytest tests/domain/test_detection.py`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implémenter**

`src/face_recognition_app/domain/detection.py` :

```python
"""
detection.py
Résultat d'une analyse : visages détectés et événement de surveillance.

Ces modèles traversent toutes les couches — moteur, bus, stockage, API, UI —
et ne dépendent donc d'aucune bibliothèque externe. Le champ `frame` est typé
`Any` pour transporter un tableau numpy sans imposer l'import à ce module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

UNKNOWN_NAME = "Inconnu"

# (haut, droite, bas, gauche) — convention de face_recognition
Location = tuple[int, int, int, int]


@dataclass
class DetectedFace:
    """Un visage détecté dans une frame."""

    location: Location
    name: str
    confidence: float
    is_known: bool

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "confidence": self.confidence,
            "is_known": self.is_known,
            "location": list(self.location),
        }

    @classmethod
    def from_dict(cls, data: dict) -> DetectedFace:
        loc = data.get("location") or (0, 0, 0, 0)
        return cls(
            location=tuple(int(v) for v in loc),  # type: ignore[arg-type]
            name=data.get("name", UNKNOWN_NAME),
            confidence=float(data.get("confidence", 0.0)),
            is_known=bool(data.get("is_known", False)),
        )


@dataclass
class SurveillanceEvent:
    """Une détection sur une caméra, à un instant donné."""

    camera_uid: str
    camera_name: str
    timestamp: float
    faces: list[DetectedFace] = field(default_factory=list)
    motion_score: float = 0.0
    frame: Any = None

    @property
    def dt(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp)

    @property
    def known_names(self) -> list[str]:
        return [f.name for f in self.faces if f.is_known]

    @property
    def unknown_count(self) -> int:
        return sum(1 for f in self.faces if not f.is_known)

    @property
    def has_unknown(self) -> bool:
        return self.unknown_count > 0
```

- [ ] **Step 4: Vérifier que les tests passent**

Run: `.venv/bin/python -m pytest tests/domain/test_detection.py`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add src/face_recognition_app/domain/detection.py tests/domain/test_detection.py
git commit -m "feat: domain/detection.py, DetectedFace et SurveillanceEvent

Modeles partages par toutes les couches, sans dependance externe.
from_dict tolere l'absence de localisation, comme les evenements
deja presents dans events.db."
```

---

### Task 6: `domain/profile.py` — profil unique de surveillance

**Files:**
- Create: `src/face_recognition_app/domain/profile.py`
- Create: `tests/domain/test_profile.py`

**Interfaces:**
- Consumes: `DetectedFace` (Task 5)
- Produces:
  - `SurveillanceProfile` avec `to_dict()`, `from_dict()`, et les décisions pures `should_alert(faces) -> bool`, `should_record() -> bool`
  - `DEFAULT_PROFILES: dict[str, SurveillanceProfile]` — clés `"present"`, `"absent"`, `"nuit"`

**Décisions de conception :**
- Les champs d'alerte (`alert_on_unknown`, `alert_on_known`, `target_persons`) vivent désormais **uniquement ici**. `AlertConfig` ne conservera que le transport (SMTP, webhook, activation des canaux).
- `enabled_camera_uids` est supprimé — champ jamais lu, conformément au registre de code mort de la spec §4.5. `from_dict` ignore la clé si elle est présente dans un `profiles.json` existant.

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/domain/test_profile.py` :

```python
import pytest

from face_recognition_app.domain.detection import UNKNOWN_NAME, DetectedFace
from face_recognition_app.domain.profile import DEFAULT_PROFILES, SurveillanceProfile


def _connu(nom="Alice"):
    return DetectedFace(location=(0, 1, 1, 0), name=nom, confidence=0.9, is_known=True)


def _inconnu():
    return DetectedFace(location=(0, 1, 1, 0), name=UNKNOWN_NAME, confidence=0.2, is_known=False)


def _profil(**kw):
    base = {"name": "test", "label": "Test"}
    base.update(kw)
    return SurveillanceProfile(**base)


def test_alerte_sur_inconnu_quand_active():
    assert _profil(alert_on_unknown=True).should_alert([_inconnu()]) is True


def test_pas_d_alerte_sur_inconnu_quand_desactive():
    assert _profil(alert_on_unknown=False).should_alert([_inconnu()]) is False


def test_pas_d_alerte_sur_connu_par_defaut():
    assert _profil(alert_on_unknown=True).should_alert([_connu()]) is False


def test_alerte_sur_connu_quand_active():
    assert _profil(alert_on_known=True).should_alert([_connu()]) is True


def test_personne_ciblee_declenche_meme_si_alertes_connues_desactivees():
    p = _profil(alert_on_unknown=False, alert_on_known=False, target_persons=["Bob"])
    assert p.should_alert([_connu("Bob")]) is True


def test_personne_non_ciblee_ne_declenche_pas():
    p = _profil(alert_on_unknown=False, alert_on_known=False, target_persons=["Bob"])
    assert p.should_alert([_connu("Alice")]) is False


def test_aucune_alerte_sans_visage():
    assert _profil(alert_on_unknown=True, alert_on_known=True).should_alert([]) is False


def test_should_record_suit_le_champ():
    assert _profil(record_video=True).should_record() is True
    assert _profil(record_video=False).should_record() is False


def test_profils_par_defaut_presents():
    assert set(DEFAULT_PROFILES) == {"present", "absent", "nuit"}


def test_profil_present_n_enregistre_pas_et_n_alerte_pas():
    p = DEFAULT_PROFILES["present"]
    assert p.should_record() is False
    assert p.should_alert([_inconnu()]) is False


def test_profil_absent_enregistre_et_alerte_sur_inconnu():
    p = DEFAULT_PROFILES["absent"]
    assert p.should_record() is True
    assert p.should_alert([_inconnu()]) is True


def test_profil_nuit_analyse_en_continu():
    assert DEFAULT_PROFILES["nuit"].motion_required is False


def test_aller_retour_dict():
    p = _profil(target_persons=["Bob"], record_video=True)
    assert SurveillanceProfile.from_dict(p.to_dict()) == p


def test_from_dict_ignore_enabled_camera_uids_obsolete():
    """profiles.json existant contient ce champ, desormais supprime."""
    p = SurveillanceProfile.from_dict(
        {"name": "x", "label": "X", "enabled_camera_uids": ["abc"]}
    )
    assert p.name == "x"
    assert not hasattr(p, "enabled_camera_uids")


def test_seuil_de_reconnaissance_invalide_rejete():
    with pytest.raises(ValueError, match="recognition_threshold"):
        _profil(recognition_threshold=1.5)


def test_intervalle_d_analyse_negatif_rejete():
    with pytest.raises(ValueError, match="analysis_interval"):
        _profil(analysis_interval=-1.0)
```

- [ ] **Step 2: Vérifier que les tests échouent**

Run: `.venv/bin/python -m pytest tests/domain/test_profile.py`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implémenter**

`src/face_recognition_app/domain/profile.py` :

```python
"""
profile.py
Profil de surveillance — source unique de vérité des règles d'analyse et d'alerte.

Les champs d'alerte vivaient jusqu'ici en double, ici et dans `AlertConfig`,
et seule la version `AlertConfig` avait un effet. Ils sont désormais définis
uniquement dans ce module ; `AlertConfig` ne conserve que le transport
(serveur SMTP, URL de webhook, activation des canaux).

Profils prédéfinis :
  present → détection légère, aucune alerte, pas d'enregistrement
  absent  → surveillance active, alertes, enregistrement
  nuit    → sensibilité maximale, analyse continue, enregistrement
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from .detection import DetectedFace

DETECTION_MODELS = frozenset({"hog", "cnn"})


@dataclass
class SurveillanceProfile:
    """Règles appliquées par le moteur de surveillance."""

    name: str
    label: str

    # Analyse
    detection_model: str = "hog"
    analysis_interval: float = 0.5
    motion_required: bool = True
    motion_sensitivity: int = 500
    recognition_threshold: float = 0.5

    # Enregistrement
    record_video: bool = False
    pre_record_seconds: float = 5.0
    post_record_seconds: float = 10.0

    # Alertes — source unique de vérité
    alert_on_unknown: bool = True
    alert_on_known: bool = False
    target_persons: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.detection_model not in DETECTION_MODELS:
            raise ValueError(
                f"detection_model invalide : {self.detection_model!r} "
                f"(attendu : {sorted(DETECTION_MODELS)})"
            )
        if not 0.0 < self.recognition_threshold <= 1.0:
            raise ValueError(
                f"recognition_threshold doit être dans ]0, 1], reçu {self.recognition_threshold}"
            )
        if self.analysis_interval < 0:
            raise ValueError(f"analysis_interval doit être positif, reçu {self.analysis_interval}")

    # ── Décisions ─────────────────────────────────────────────────────────────

    def should_alert(self, faces: Sequence[DetectedFace]) -> bool:
        """Faut-il émettre une alerte pour ces visages ?"""
        if not faces:
            return False
        known = [f for f in faces if f.is_known]
        if any(f.name in self.target_persons for f in known):
            return True
        if self.alert_on_unknown and any(not f.is_known for f in faces):
            return True
        return bool(self.alert_on_known and known)

    def should_record(self) -> bool:
        """Faut-il enregistrer un clip vidéo sur détection ?"""
        return self.record_video

    # ── Sérialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> SurveillanceProfile:
        """Ignore les clés inconnues, dont `enabled_camera_uids` désormais supprimé."""
        champs = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in champs})


DEFAULT_PROFILES: dict[str, SurveillanceProfile] = {
    "present": SurveillanceProfile(
        name="present",
        label="Présent (domicile occupé)",
        analysis_interval=1.0,
        motion_sensitivity=800,
        record_video=False,
        alert_on_unknown=False,
        alert_on_known=False,
    ),
    "absent": SurveillanceProfile(
        name="absent",
        label="Absent (surveillance active)",
        analysis_interval=0.5,
        motion_sensitivity=400,
        record_video=True,
        post_record_seconds=15.0,
        alert_on_unknown=True,
        alert_on_known=True,
    ),
    "nuit": SurveillanceProfile(
        name="nuit",
        label="Nuit (sensibilité maximale)",
        analysis_interval=0.3,
        motion_required=False,
        motion_sensitivity=200,
        recognition_threshold=0.45,
        record_video=True,
        pre_record_seconds=10.0,
        post_record_seconds=20.0,
        alert_on_unknown=True,
        alert_on_known=False,
    ),
}
```

- [ ] **Step 4: Vérifier que les tests passent**

Run: `.venv/bin/python -m pytest tests/domain/test_profile.py`
Expected: `16 passed`

- [ ] **Step 5: Vérifier que le vrai `profiles.json` se relit**

Run: `.venv/bin/python -c "
import json, sys; sys.path.insert(0,'src')
from face_recognition_app.domain.profile import SurveillanceProfile
data = json.loads(open('profiles.json').read())
for d in data['profiles']:
    p = SurveillanceProfile.from_dict(d)
    print(f'{p.name}: enregistre={p.should_record()}')
"`
Expected: trois lignes, `present: enregistre=False`, `absent: enregistre=True`, `nuit: enregistre=True`

- [ ] **Step 6: Commit**

```bash
git add src/face_recognition_app/domain/profile.py tests/domain/test_profile.py
git commit -m "feat: domain/profile.py, profil unique avec decisions pures

Fusionne les champs d'alerte qui vivaient en double avec AlertConfig,
ou seule la version AlertConfig avait un effet.

should_alert et should_record rendent testable la logique qui etait
dispersee dans le moteur. Supprime enabled_camera_uids, jamais lu.
Ajoute la validation des seuils, absente jusqu'ici."
```

---

### Task 7: `storage/encodings_repository.py` — encodages injectés

**Files:**
- Create: `src/face_recognition_app/storage/encodings_repository.py`
- Create: `tests/storage/test_encodings_repository.py`
- Modify: `src/face_recognition_app/storage/encodings_store.py` (devient un adaptateur)
- Delete: `tests/test_encodings_store.py`

**Interfaces:**
- Consumes: `AppSettings` (Task 3), `find_duplicate` (Task 2)
- Produces: `EncodingsRepository(settings: AppSettings)` avec :
  - `load_all() -> list[StoredEncoding]` (`StoredEncoding` = dataclass `uid`, `name`, `encoding`, `timestamp`)
  - `load_map() -> dict[str, np.ndarray]`
  - `save(name: str, encoding: np.ndarray, image: np.ndarray | None = None) -> str` (retourne l'uid)
  - `load_image(name: str) -> np.ndarray | None`
  - `delete(name: str) -> list[str]`
  - `load_metadata() -> dict`
  - `find_duplicate_name(encoding: np.ndarray, tolerance: float | None = None) -> str | None`

L'ancien module `encodings_store` conserve ses fonctions publiques, qui délèguent à un repository construit sur `AppSettings.create()`, afin que `services/` et `ui/` continuent de fonctionner jusqu'aux Phases 2 et 3.

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/storage/test_encodings_repository.py` :

```python
import numpy as np
import pytest

from face_recognition_app.settings import AppSettings
from face_recognition_app.storage.encodings_repository import EncodingsRepository


@pytest.fixture
def repo(tmp_path):
    settings = AppSettings.create(project_root=tmp_path)
    settings.ensure_directories()
    return EncodingsRepository(settings)


def test_sauvegarde_puis_relecture(repo):
    uid = repo.save("Alice", np.zeros(128))
    assert len(uid) == 12
    assert [e.name for e in repo.load_all()] == ["Alice"]


def test_sauvegarde_avec_image_relisible(repo):
    repo.save("Alice", np.zeros(128), image=np.zeros((50, 50, 3), dtype=np.uint8))
    assert repo.load_image("Alice") is not None


def test_sans_image_load_image_retourne_none(repo):
    repo.save("Bob", np.ones(128))
    assert repo.load_image("Bob") is None


def test_suppression(repo):
    repo.save("Alice", np.zeros(128))
    assert repo.delete("Alice")
    assert repo.load_all() == []


def test_suppression_inexistant_retourne_liste_vide(repo):
    assert repo.delete("Personne") == []


def test_load_map_dedoublonne_les_noms(repo):
    repo.save("Alice", np.zeros(128))
    repo.save("Alice", np.zeros(128))
    assert list(repo.load_map()) == ["Alice"]


def test_load_map_retourne_des_vecteurs_128(repo):
    repo.save("Alice", np.zeros(128))
    assert repo.load_map()["Alice"].shape == (128,)


def test_metadata_mise_a_jour_a_la_sauvegarde(repo):
    repo.save("Charlie", np.zeros(128))
    assert "Charlie" in [v["name"] for v in repo.load_metadata().values()]


def test_metadata_nettoyee_a_la_suppression(repo):
    repo.save("Dave", np.zeros(128))
    repo.delete("Dave")
    assert "Dave" not in [v["name"] for v in repo.load_metadata().values()]


def test_fichier_corrompu_ignore_sans_planter(repo, tmp_path):
    repo.save("Alice", np.zeros(128))
    (tmp_path / "encodings" / "corrompu.json").write_text("{ pas du json")
    assert [e.name for e in repo.load_all()] == ["Alice"]


def test_encodage_de_mauvaise_longueur_ignore(repo, tmp_path):
    import json

    repo.save("Alice", np.zeros(128))
    (tmp_path / "encodings" / "court.json").write_text(
        json.dumps({"name": "Court", "encoding": [0.0] * 64, "timestamp": "2026-01-01T00:00:00"})
    )
    assert [e.name for e in repo.load_all()] == ["Alice"]


def test_find_duplicate_name(repo):
    vec = np.full(128, 0.1)
    repo.save("Alice", vec)
    assert repo.find_duplicate_name(vec) == "Alice"
    assert repo.find_duplicate_name(np.ones(128)) is None


def test_permissions_restrictives_sur_les_encodages(repo, tmp_path):
    """Donnees biometriques : lisibles par le seul proprietaire."""
    repo.save("Alice", np.zeros(128))
    fichier = next(p for p in (tmp_path / "encodings").glob("*.json") if p.name != "metadata.json")
    assert fichier.stat().st_mode & 0o077 == 0
```

- [ ] **Step 2: Vérifier que les tests échouent**

Run: `.venv/bin/python -m pytest tests/storage/test_encodings_repository.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'face_recognition_app.storage.encodings_repository'`

- [ ] **Step 3: Implémenter le repository**

`src/face_recognition_app/storage/encodings_repository.py` :

```python
"""
encodings_repository.py
Stockage des encodages faciaux — un fichier JSON par visage.

Format d'un fichier (inchangé par rapport à la version précédente) :
    {"name": str, "encoding": [128 flottants], "timestamp": iso, "image_base64": str?}

`metadata.json` indexe les uid vers nom et date de création.

Les fichiers sont écrits en 0600 : ce sont des données biométriques.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ..domain.matching import ENCODING_LENGTH, find_duplicate
from ..settings import AppSettings

logger = logging.getLogger(__name__)


@dataclass
class StoredEncoding:
    """Un encodage lu depuis le disque."""

    uid: str
    name: str
    encoding: np.ndarray
    timestamp: str


class EncodingsRepository:
    """Accès aux encodages faciaux persistés."""

    def __init__(self, settings: AppSettings) -> None:
        self._dir = settings.encodings_dir
        self._meta_file = settings.encodings_dir / "metadata.json"
        self._default_tolerance = settings.duplicate_tolerance

    # ── Lecture ───────────────────────────────────────────────────────────────

    def _iter_files(self) -> Iterator[Path]:
        if not self._dir.exists():
            return
        for path in sorted(self._dir.glob("*.json")):
            if path.name == self._meta_file.name or "temp" in path.name:
                continue
            yield path

    def _read_json(self, path: Path) -> dict | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Encodage illisible ignoré (%s) : %s", path.name, exc)
            return None

    @staticmethod
    def _is_valid(data: dict) -> bool:
        if not all(k in data for k in ("name", "encoding", "timestamp")):
            return False
        return isinstance(data["encoding"], list) and len(data["encoding"]) == ENCODING_LENGTH

    def load_all(self) -> list[StoredEncoding]:
        """Tous les encodages valides, triés par nom de fichier."""
        resultats: list[StoredEncoding] = []
        for path in self._iter_files():
            data = self._read_json(path)
            if data is None or not self._is_valid(data):
                if data is not None:
                    logger.warning("Encodage invalide ignoré : %s", path.name)
                continue
            resultats.append(
                StoredEncoding(
                    uid=path.stem,
                    name=data["name"],
                    encoding=np.array(data["encoding"], dtype=float),
                    timestamp=data["timestamp"],
                )
            )
        return resultats

    def load_map(self) -> dict[str, np.ndarray]:
        """Un encodage par nom — le premier rencontré en cas de doublon."""
        mapping: dict[str, np.ndarray] = {}
        for entry in self.load_all():
            mapping.setdefault(entry.name, entry.encoding)
        return mapping

    def load_metadata(self) -> dict:
        data = self._read_json(self._meta_file)
        return data if isinstance(data, dict) else {}

    def load_image(self, name: str) -> np.ndarray | None:
        """Image du visage enregistré pour ce nom, ou None."""
        for path in self._iter_files():
            data = self._read_json(path)
            if data is None or data.get("name") != name:
                continue
            encoded = data.get("image_base64") or data.get("image")
            if isinstance(encoded, str):
                return self._decode_image(encoded)
        return None

    @staticmethod
    def _decode_image(encoded: str) -> np.ndarray | None:
        try:
            raw = base64.b64decode(encoded)
            return cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception as exc:
            logger.warning("Image d'encodage indécodable : %s", exc)
            return None

    # ── Écriture ──────────────────────────────────────────────────────────────

    def _write_json(self, path: Path, data: dict) -> None:
        path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")
        os.chmod(path, 0o600)

    def save(
        self,
        name: str,
        encoding: np.ndarray,
        image: np.ndarray | None = None,
    ) -> str:
        """Enregistre un encodage et retourne son uid."""
        encoding = np.asarray(encoding, dtype=float)
        if encoding.shape != (ENCODING_LENGTH,):
            raise ValueError(
                f"Un encodage doit être un vecteur de {ENCODING_LENGTH} valeurs, "
                f"reçu {encoding.shape}"
            )

        timestamp = datetime.now().isoformat()
        uid = uuid.uuid4().hex[:12]
        data: dict = {
            "name": name,
            "encoding": encoding.tolist(),
            "timestamp": timestamp,
        }

        if image is not None:
            ok, buffer = cv2.imencode(".jpg", image)
            if ok:
                data["image_base64"] = base64.b64encode(buffer).decode("ascii")
            else:
                logger.warning("Encodage JPEG échoué pour '%s' ; visage enregistré sans image", name)

        self._dir.mkdir(parents=True, exist_ok=True)
        self._write_json(self._dir / f"{uid}.json", data)
        self._update_metadata(uid, name, timestamp)
        logger.info("Visage '%s' enregistré (uid=%s)", name, uid)
        return uid

    def _update_metadata(self, uid: str, name: str, timestamp: str) -> None:
        metadata = self.load_metadata()
        metadata[uid] = {"name": name, "date_creation": timestamp}
        self._write_json(self._meta_file, metadata)

    def delete(self, name: str) -> list[str]:
        """Supprime tous les encodages portant ce nom. Retourne les uid supprimés."""
        supprimes: list[str] = []
        for path in self._iter_files():
            data = self._read_json(path)
            if data is None or data.get("name") != name:
                continue
            try:
                path.unlink()
                supprimes.append(path.stem)
            except OSError as exc:
                logger.error("Suppression impossible (%s) : %s", path.name, exc)

        if supprimes:
            metadata = self.load_metadata()
            for uid in supprimes:
                metadata.pop(uid, None)
            self._write_json(self._meta_file, metadata)
            logger.info("Visage '%s' supprimé (%d fichier(s))", name, len(supprimes))
        return supprimes

    # ── Doublons ──────────────────────────────────────────────────────────────

    def find_duplicate_name(
        self,
        encoding: np.ndarray,
        tolerance: float | None = None,
    ) -> str | None:
        """Nom de la personne déjà enregistrée correspondant à cet encodage, ou None."""
        existing = [(e.name, e.encoding) for e in self.load_all()]
        return find_duplicate(
            encoding,
            existing,
            self._default_tolerance if tolerance is None else tolerance,
        )
```

- [ ] **Step 4: Vérifier que les tests passent**

Run: `.venv/bin/python -m pytest tests/storage/test_encodings_repository.py`
Expected: `13 passed`

- [ ] **Step 5: Transformer l'ancien module en adaptateur**

Remplacer intégralement `src/face_recognition_app/storage/encodings_store.py` :

```python
"""
encodings_store.py
ADAPTATEUR TEMPORAIRE — conserve l'API par fonctions utilisée par services/ et ui/.

Délègue à EncodingsRepository. Sera supprimé en Phase 3, quand tous les appelants
auront reçu le repository par injection.
"""

from __future__ import annotations

import numpy as np

from ..settings import AppSettings
from .encodings_repository import EncodingsRepository

_repo: EncodingsRepository | None = None


def _repository() -> EncodingsRepository:
    global _repo
    if _repo is None:
        settings = AppSettings.create()
        settings.ensure_directories()
        _repo = EncodingsRepository(settings)
    return _repo


def set_repository(repo: EncodingsRepository) -> None:
    """Injecté par __main__ pour que l'adaptateur partage le repository de l'application."""
    global _repo
    _repo = repo


def load_existing_encodings() -> list[dict]:
    return [{"name": e.name, "encoding": e.encoding} for e in _repository().load_all()]


def load_encodings_map() -> dict[str, np.ndarray]:
    return _repository().load_map()


def load_metadata() -> dict:
    return _repository().load_metadata()


def save_face_encoding(name, encoding, image=None) -> None:
    _repository().save(name, np.asarray(encoding, dtype=float), image)


def load_image_for_name(name: str):
    return _repository().load_image(name)


def delete_encoding(name: str) -> list[str]:
    return _repository().delete(name)
```

- [ ] **Step 6: Supprimer l'ancien fichier de tests**

Ses cas sont couverts par `tests/storage/test_encodings_repository.py` (encodages) et `tests/domain/test_matching.py` (doublons), sans mutation de constantes de module.

```bash
rm tests/test_encodings_store.py
```

- [ ] **Step 7: Vérifier la suite complète et l'application**

Run: `.venv/bin/python -m pytest`
Expected: `61 passed` (11 matching + 6 settings + 8 camera + 7 detection + 16 profile + 13 encodings ; les 14 tests de `test_encodings_store.py` viennent d'être supprimés)

Run: `.venv/bin/python -c "
import sys; sys.path.insert(0,'src')
from face_recognition_app.storage.encodings_store import load_encodings_map
print(sorted(load_encodings_map()))
"`
Expected: `['ravaka', 'vic', 'victorien']`

- [ ] **Step 8: Commit**

```bash
git add -A src/face_recognition_app/storage tests
git commit -m "refactor: EncodingsRepository injecte, encodings_store devient adaptateur

Le repository recoit AppSettings au lieu de lire des constantes de module,
ce qui supprime la mutation de ENCODED_DIR dans les tests.

Ajoute : validation de la longueur d'encodage a l'ecriture, permissions
0600 sur les donnees biometriques, journalisation des fichiers ignores
au lieu d'un echec silencieux."
```

---

### Task 8: `storage/event_repository.py` — événements, écriture asynchrone et purge

**Files:**
- Create: `src/face_recognition_app/storage/event_repository.py`
- Create: `tests/storage/test_event_repository.py`
- Modify: `src/face_recognition_app/storage/event_store.py` (devient un adaptateur)

**Interfaces:**
- Consumes: `AppSettings` (Task 3), `DetectedFace` / `SurveillanceEvent` (Task 5)
- Produces: `EventRepository(settings: AppSettings)` avec :
  - `start() -> None` / `stop() -> None` (thread d'écriture ; idempotents)
  - `record(timestamp, camera_uid, camera_name, faces, frame=None, save_snapshot=True) -> None` — non bloquant
  - `record_sync(...) -> StoredEvent` — écriture immédiate, utilisée par les tests
  - `get_recent(count=100)`, `get_by_id(event_id)`, `get_by_camera(uid, count=100)`, `get_by_person(name, count=100)`
  - `count() -> int`, `stats() -> dict`
  - `delete_before(timestamp) -> int`, `purge_expired() -> int`, `vacuum() -> None`
  - `StoredEvent` : `id`, `timestamp`, `camera_uid`, `camera_name`, `faces: list[DetectedFace]`, `snapshot_b64`, propriétés `dt`, `known_names`, `unknown_count`, `has_unknown`, `to_dict()`

Le schéma SQLite `events(id, timestamp, camera_uid, camera_name, faces_json, snapshot_b64)` **reste inchangé** : `events.db` existant est lu sans migration.

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/storage/test_event_repository.py` :

```python
import time

import numpy as np
import pytest

from face_recognition_app.domain.detection import UNKNOWN_NAME, DetectedFace
from face_recognition_app.settings import AppSettings
from face_recognition_app.storage.event_repository import EventRepository


@pytest.fixture
def repo(tmp_path):
    settings = AppSettings.create(project_root=tmp_path)
    settings.ensure_directories()
    r = EventRepository(settings)
    yield r
    r.stop()


def _connu(nom="Alice"):
    return DetectedFace(location=(0, 10, 10, 0), name=nom, confidence=0.9, is_known=True)


def _inconnu():
    return DetectedFace(location=(0, 10, 10, 0), name=UNKNOWN_NAME, confidence=0.1, is_known=False)


def test_enregistrement_et_relecture(repo):
    repo.record_sync(1000.0, "cam1", "Salon", [_connu()])
    events = repo.get_recent()
    assert len(events) == 1
    assert events[0].camera_name == "Salon"
    assert events[0].known_names == ["Alice"]


def test_tri_du_plus_recent_au_plus_ancien(repo):
    repo.record_sync(1000.0, "cam1", "Salon", [])
    repo.record_sync(2000.0, "cam1", "Salon", [])
    assert [e.timestamp for e in repo.get_recent()] == [2000.0, 1000.0]


def test_limite_respectee(repo):
    for i in range(5):
        repo.record_sync(1000.0 + i, "cam1", "Salon", [])
    assert len(repo.get_recent(count=3)) == 3


def test_get_by_id(repo):
    evt = repo.record_sync(1000.0, "cam1", "Salon", [_connu()])
    assert repo.get_by_id(evt.id).camera_name == "Salon"


def test_get_by_id_inexistant(repo):
    assert repo.get_by_id(999) is None


def test_filtre_par_camera(repo):
    repo.record_sync(1000.0, "cam1", "Salon", [])
    repo.record_sync(1001.0, "cam2", "Cuisine", [])
    assert [e.camera_name for e in repo.get_by_camera("cam2")] == ["Cuisine"]


def test_filtre_par_personne(repo):
    repo.record_sync(1000.0, "cam1", "Salon", [_connu("Alice")])
    repo.record_sync(1001.0, "cam1", "Salon", [_connu("Bob")])
    assert [e.known_names for e in repo.get_by_person("Bob")] == [["Bob"]]


def test_filtre_par_personne_echappe_les_jokers_like(repo):
    """Un nom contenant % ne doit pas se comporter comme un joker."""
    repo.record_sync(1000.0, "cam1", "Salon", [_connu("Alice")])
    assert repo.get_by_person("%") == []


def test_comptage(repo):
    repo.record_sync(1000.0, "cam1", "Salon", [])
    repo.record_sync(1001.0, "cam1", "Salon", [])
    assert repo.count() == 2


def test_statistiques_par_camera(repo):
    repo.record_sync(1000.0, "cam1", "Salon", [])
    repo.record_sync(1001.0, "cam2", "Cuisine", [])
    repo.record_sync(1002.0, "cam2", "Cuisine", [])
    stats = repo.stats()
    assert stats["total"] == 3
    assert stats["by_camera"]["Cuisine"] == 2


def test_snapshot_encode_et_redimensionne(repo):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    evt = repo.record_sync(1000.0, "cam1", "Salon", [_inconnu()], frame=frame)
    assert evt.snapshot_b64 is not None
    assert len(evt.snapshot_b64) > 0


def test_pas_de_snapshot_si_desactive(repo):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    evt = repo.record_sync(1000.0, "cam1", "Salon", [], frame=frame, save_snapshot=False)
    assert evt.snapshot_b64 is None


def test_suppression_avant_une_date(repo):
    repo.record_sync(1000.0, "cam1", "Salon", [])
    repo.record_sync(3000.0, "cam1", "Salon", [])
    assert repo.delete_before(2000.0) == 1
    assert repo.count() == 1


def test_purge_selon_la_retention(tmp_path):
    settings = AppSettings.create(project_root=tmp_path)
    settings.ensure_directories()
    repo = EventRepository(settings)
    try:
        ancien = time.time() - (settings.event_retention_days + 1) * 86400
        repo.record_sync(ancien, "cam1", "Salon", [])
        repo.record_sync(time.time(), "cam1", "Salon", [])
        assert repo.purge_expired() == 1
        assert repo.count() == 1
    finally:
        repo.stop()


def test_ecriture_asynchrone_finit_par_persister(repo):
    repo.start()
    repo.record(1000.0, "cam1", "Salon", [_connu()])
    repo.stop()  # draine la file avant de rendre la main
    assert repo.count() == 1


def test_stop_est_idempotent(repo):
    repo.start()
    repo.stop()
    repo.stop()


def test_visages_relus_comme_objets_du_domaine(repo):
    repo.record_sync(1000.0, "cam1", "Salon", [_connu("Alice"), _inconnu()])
    evt = repo.get_recent()[0]
    assert all(isinstance(f, DetectedFace) for f in evt.faces)
    assert evt.unknown_count == 1
    assert evt.has_unknown is True
```

- [ ] **Step 2: Vérifier que les tests échouent**

Run: `.venv/bin/python -m pytest tests/storage/test_event_repository.py`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implémenter le repository**

`src/face_recognition_app/storage/event_repository.py` :

```python
"""
event_repository.py
Journal persistant des événements de surveillance — SQLite.

Deux améliorations par rapport à la version précédente :

  - **Écriture asynchrone.** `record()` dépose l'événement dans une file et rend
    la main immédiatement ; un thread dédié encode le snapshot et écrit en base.
    Auparavant l'encodage JPEG et l'INSERT se faisaient dans le thread d'analyse,
    sous verrou global, ce qui sérialisait les caméras.
  - **Purge automatique.** `purge_expired()` applique `settings.event_retention_days`,
    empêchant la base de croître sans fin.

Schéma inchangé, `events.db` existant est lu sans migration :
    events(id, timestamp, camera_uid, camera_name, faces_json, snapshot_b64)
"""

from __future__ import annotations

import base64
import json
import logging
import os
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..domain.detection import DetectedFace
from ..settings import AppSettings

logger = logging.getLogger(__name__)

_SENTINELLE = object()


@dataclass
class StoredEvent:
    """Un événement lu depuis la base."""

    id: int
    timestamp: float
    camera_uid: str
    camera_name: str
    faces: list[DetectedFace] = field(default_factory=list)
    snapshot_b64: str | None = None

    @property
    def dt(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp)

    @property
    def known_names(self) -> list[str]:
        return [f.name for f in self.faces if f.is_known]

    @property
    def unknown_count(self) -> int:
        return sum(1 for f in self.faces if not f.is_known)

    @property
    def has_unknown(self) -> bool:
        return self.unknown_count > 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "datetime": self.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "camera_uid": self.camera_uid,
            "camera_name": self.camera_name,
            "faces": [f.to_dict() for f in self.faces],
            "snapshot_b64": self.snapshot_b64,
        }


def _escape_like(value: str) -> str:
    """Neutralise les jokers LIKE dans une valeur fournie par l'utilisateur."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class EventRepository:
    """Accès aux événements de surveillance persistés."""

    SNAPSHOT_WIDTH = 320
    SNAPSHOT_QUALITY = 60
    DEFAULT_COUNT = 100
    QUEUE_MAXSIZE = 256

    def __init__(self, settings: AppSettings) -> None:
        self._db_path = settings.events_db
        self._retention_days = settings.event_retention_days
        self._local = threading.local()
        self._connections: list[sqlite3.Connection] = []
        self._conn_lock = threading.Lock()
        self._write_lock = threading.Lock()

        self._queue: queue.Queue = queue.Queue(maxsize=self.QUEUE_MAXSIZE)
        self._writer: threading.Thread | None = None
        self._running = False

        self._init_db()

    # ── Connexions ────────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
            with self._conn_lock:
                self._connections.append(conn)
        return conn

    def _init_db(self) -> None:
        nouveau = not self._db_path.exists()
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp    REAL    NOT NULL,
                    camera_uid   TEXT    NOT NULL,
                    camera_name  TEXT    NOT NULL,
                    faces_json   TEXT    NOT NULL,
                    snapshot_b64 TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ts  ON events(timestamp DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cam ON events(camera_uid)")
            conn.commit()
        finally:
            conn.close()
        if nouveau:
            os.chmod(self._db_path, 0o600)

    def close_connections(self) -> None:
        """Ferme toutes les connexions ouvertes. Appelé à l'arrêt de l'application."""
        with self._conn_lock:
            for conn in self._connections:
                try:
                    conn.close()
                except sqlite3.Error as exc:
                    logger.warning("Fermeture de connexion SQLite échouée : %s", exc)
            self._connections.clear()
        self._local = threading.local()

    # ── Thread d'écriture ─────────────────────────────────────────────────────

    def start(self) -> None:
        """Démarre le thread d'écriture. Idempotent."""
        if self._running:
            return
        self._running = True
        self._writer = threading.Thread(target=self._write_loop, name="event-writer", daemon=True)
        self._writer.start()
        logger.info("EventRepository démarré (rétention %d jours)", self._retention_days)

    def stop(self) -> None:
        """Draine la file, arrête le thread et ferme les connexions. Idempotent."""
        if self._running:
            self._running = False
            self._queue.put(_SENTINELLE)
            if self._writer is not None:
                self._writer.join(timeout=5.0)
            self._writer = None
        self.close_connections()

    def _write_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SENTINELLE:
                break
            try:
                self.record_sync(**item)
            except Exception as exc:
                logger.error("Écriture d'événement échouée : %s", exc, exc_info=True)

    # ── Écriture ──────────────────────────────────────────────────────────────

    def record(
        self,
        timestamp: float,
        camera_uid: str,
        camera_name: str,
        faces: list[DetectedFace],
        frame: np.ndarray | None = None,
        save_snapshot: bool = True,
    ) -> None:
        """Dépose l'événement dans la file d'écriture. Ne bloque jamais l'appelant."""
        item = {
            "timestamp": timestamp,
            "camera_uid": camera_uid,
            "camera_name": camera_name,
            "faces": faces,
            "frame": frame,
            "save_snapshot": save_snapshot,
        }
        if not self._running:
            self.record_sync(**item)
            return
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            logger.warning("File d'écriture saturée, événement %s abandonné", camera_name)

    def record_sync(
        self,
        timestamp: float,
        camera_uid: str,
        camera_name: str,
        faces: list[DetectedFace],
        frame: np.ndarray | None = None,
        save_snapshot: bool = True,
    ) -> StoredEvent:
        """Écrit immédiatement l'événement et le retourne."""
        snapshot = self._encode_snapshot(frame) if (frame is not None and save_snapshot) else None
        faces_json = json.dumps([f.to_dict() for f in faces], ensure_ascii=False)

        with self._write_lock:
            conn = self._conn()
            cur = conn.execute(
                "INSERT INTO events(timestamp, camera_uid, camera_name, faces_json, snapshot_b64) "
                "VALUES (?, ?, ?, ?, ?)",
                (timestamp, camera_uid, camera_name, faces_json, snapshot),
            )
            conn.commit()
            event_id = int(cur.lastrowid or 0)

        return StoredEvent(
            id=event_id,
            timestamp=timestamp,
            camera_uid=camera_uid,
            camera_name=camera_name,
            faces=list(faces),
            snapshot_b64=snapshot,
        )

    def _encode_snapshot(self, frame: np.ndarray) -> str | None:
        try:
            h, w = frame.shape[:2]
            if w > self.SNAPSHOT_WIDTH:
                scale = self.SNAPSHOT_WIDTH / w
                frame = cv2.resize(frame, (self.SNAPSHOT_WIDTH, int(h * scale)))
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.SNAPSHOT_QUALITY])
            if ok:
                return base64.b64encode(buf.tobytes()).decode("ascii")
            logger.warning("Encodage JPEG du snapshot échoué")
        except Exception as exc:
            logger.error("Erreur d'encodage du snapshot : %s", exc)
        return None

    # ── Lecture ───────────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> StoredEvent:
        return StoredEvent(
            id=row["id"],
            timestamp=row["timestamp"],
            camera_uid=row["camera_uid"],
            camera_name=row["camera_name"],
            faces=[DetectedFace.from_dict(d) for d in json.loads(row["faces_json"])],
            snapshot_b64=row["snapshot_b64"],
        )

    def _query(self, sql: str, params: tuple[Any, ...]) -> list[StoredEvent]:
        rows = self._conn().execute(sql, params).fetchall()
        return [self._row_to_event(r) for r in rows]

    def get_recent(self, count: int = DEFAULT_COUNT) -> list[StoredEvent]:
        return self._query("SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (count,))

    def get_by_id(self, event_id: int) -> StoredEvent | None:
        row = self._conn().execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        return self._row_to_event(row) if row else None

    def get_by_camera(self, camera_uid: str, count: int = DEFAULT_COUNT) -> list[StoredEvent]:
        return self._query(
            "SELECT * FROM events WHERE camera_uid = ? ORDER BY timestamp DESC LIMIT ?",
            (camera_uid, count),
        )

    def get_by_person(self, name: str, count: int = DEFAULT_COUNT) -> list[StoredEvent]:
        """Recherche par nom dans le JSON des visages, jokers LIKE neutralisés."""
        pattern = f'%"name": "{_escape_like(name)}"%'
        return self._query(
            "SELECT * FROM events WHERE faces_json LIKE ? ESCAPE '\\' "
            "ORDER BY timestamp DESC LIMIT ?",
            (pattern, count),
        )

    def count(self) -> int:
        return int(self._conn().execute("SELECT COUNT(*) FROM events").fetchone()[0])

    def stats(self) -> dict:
        conn = self._conn()
        total = int(conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])
        by_camera = {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT camera_name, COUNT(*) FROM events GROUP BY camera_name"
            ).fetchall()
        }
        return {"total": total, "by_camera": by_camera}

    # ── Nettoyage ─────────────────────────────────────────────────────────────

    def delete_before(self, before_timestamp: float) -> int:
        with self._write_lock:
            conn = self._conn()
            cur = conn.execute("DELETE FROM events WHERE timestamp < ?", (before_timestamp,))
            conn.commit()
            return cur.rowcount

    def purge_expired(self) -> int:
        """Supprime les événements plus vieux que la rétention configurée."""
        cutoff = time.time() - self._retention_days * 86400
        supprimes = self.delete_before(cutoff)
        if supprimes:
            logger.info("Purge : %d événement(s) au-delà de %d jours supprimé(s)",
                        supprimes, self._retention_days)
        return supprimes

    def vacuum(self) -> None:
        """Compacte la base après une purge importante."""
        with self._write_lock:
            self._conn().execute("VACUUM")
```

- [ ] **Step 4: Vérifier que les tests passent**

Run: `.venv/bin/python -m pytest tests/storage/test_event_repository.py`
Expected: `17 passed`

- [ ] **Step 5: Transformer l'ancien module en adaptateur**

Remplacer intégralement `src/face_recognition_app/storage/event_store.py` :

```python
"""
event_store.py
ADAPTATEUR TEMPORAIRE — conserve l'API utilisée par ui/ et services/.

`EventStore` délègue à `EventRepository`. Les visages sont acceptés sous forme
de dictionnaires (ancienne API) comme d'objets `DetectedFace`.

Sera supprimé en Phase 3.
"""

from __future__ import annotations

from ..domain.detection import DetectedFace
from ..settings import AppSettings
from .event_repository import EventRepository, StoredEvent

__all__ = ["EventStore", "StoredEvent"]


class EventStore:
    """Façade de compatibilité au-dessus d'EventRepository."""

    def __init__(self, db_path=None) -> None:
        settings = AppSettings.create()
        settings.ensure_directories()
        self._repo = EventRepository(settings)

    @property
    def repository(self) -> EventRepository:
        return self._repo

    @staticmethod
    def _to_faces(faces) -> list[DetectedFace]:
        return [f if isinstance(f, DetectedFace) else DetectedFace.from_dict(f) for f in faces]

    def record(self, timestamp, camera_uid, camera_name, faces, frame=None, save_snapshot=True):
        return self._repo.record_sync(
            timestamp, camera_uid, camera_name, self._to_faces(faces), frame, save_snapshot
        )

    def get_recent(self, count=100):
        return self._repo.get_recent(count)

    def get_by_id(self, event_id):
        return self._repo.get_by_id(event_id)

    def get_by_camera(self, camera_uid, count=100):
        return self._repo.get_by_camera(camera_uid, count)

    def get_by_person(self, name, count=100):
        return self._repo.get_by_person(name, count)

    def count(self):
        return self._repo.count()

    def stats(self):
        return self._repo.stats()

    def delete_before(self, before_timestamp):
        return self._repo.delete_before(before_timestamp)
```

**Attention pour l'appelant existant :** `ui/event_browser.py` et `ui/surveillance_dashboard.py` lisent `f.get("is_known")` sur les visages. `StoredEvent.faces` contient désormais des `DetectedFace`. Adapter les trois emplacements :

- `ui/event_browser.py:146` — `sum(1 for f in evt.faces if not f.get("is_known", False))` devient `evt.unknown_count`
- `ui/event_browser.py:202-205` — `f.get("is_known")`, `f.get("confidence", 0)`, `f.get("name", "?")` deviennent `f.is_known`, `f.confidence`, `f.name`
- `ui/surveillance_dashboard.py:460-463` — la construction de `faces_data` en dictionnaires devient `faces=event.faces`

- [ ] **Step 6: Appliquer les adaptations d'appelants**

Dans `src/face_recognition_app/ui/event_browser.py`, remplacer dans `_refresh_tree` :

```python
            unknown_n = sum(1 for f in evt.faces if not f.get("is_known", False))
```

par :

```python
            unknown_n = evt.unknown_count
```

et dans `_show_event_detail`, remplacer la boucle :

```python
        for f in evt.faces:
            status = "✓ Connu" if f.get("is_known") else "✗ Inconnu"
            conf = f.get("confidence", 0)
            lines.append(f"  {f.get('name', '?')} — {status} ({conf:.0%})")
```

par :

```python
        for f in evt.faces:
            status = "✓ Connu" if f.is_known else "✗ Inconnu"
            lines.append(f"  {f.name} — {status} ({f.confidence:.0%})")
```

Dans `src/face_recognition_app/ui/surveillance_dashboard.py`, dans `_on_surveillance_event`, remplacer :

```python
        faces_data = [
            {"name": f.name, "confidence": f.confidence, "is_known": f.is_known}
            for f in event.faces
        ]
        self._event_store.record(
            timestamp=event.timestamp,
            camera_uid=event.camera_uid,
            camera_name=event.camera_name,
            faces=faces_data,
            frame=event.frame,
            save_snapshot=True,
        )
```

par :

```python
        self._event_store.record(
            timestamp=event.timestamp,
            camera_uid=event.camera_uid,
            camera_name=event.camera_name,
            faces=event.faces,
            frame=event.frame,
            save_snapshot=True,
        )
```

- [ ] **Step 7: Vérifier que la vraie base se relit**

Run: `.venv/bin/python -c "
import sys; sys.path.insert(0,'src')
from face_recognition_app.storage.event_store import EventStore
s = EventStore()
print('total:', s.count())
e = s.get_recent(1)
if e:
    print('dernier:', e[0].dt, e[0].camera_name, e[0].known_names, 'inconnus:', e[0].unknown_count)
s.repository.stop()
"`
Expected: un total non nul et une ligne décrivant le dernier événement, sans exception.

- [ ] **Step 8: Vérifier la suite complète**

Run: `.venv/bin/python -m pytest`
Expected: `78 passed`

- [ ] **Step 9: Commit**

```bash
git add -A src/face_recognition_app tests
git commit -m "refactor: EventRepository, ecriture asynchrone et purge automatique

record() depose dans une file et rend la main : l'encodage JPEG et
l'INSERT ne bloquent plus le thread d'analyse, qui les executait
jusqu'ici sous verrou global.

Ajoute purge_expired() selon settings.event_retention_days, la
fermeture des connexions thread-local qui fuyaient a chaque cycle
start/stop, et l'echappement des jokers LIKE dans get_by_person.

Les visages sont desormais des DetectedFace et non des dictionnaires ;
event_browser et surveillance_dashboard adaptes en consequence."
```

---

### Task 9: `storage/profile_repository.py` — profils et migration des alertes

**Files:**
- Create: `src/face_recognition_app/storage/profile_repository.py`
- Create: `tests/storage/test_profile_repository.py`
- Modify: `src/face_recognition_app/storage/profile_store.py` (devient un adaptateur)

**Interfaces:**
- Consumes: `AppSettings` (Task 3), `SurveillanceProfile` / `DEFAULT_PROFILES` (Task 6)
- Produces: `ProfileRepository(settings: AppSettings)` avec :
  - `list_all() -> list[SurveillanceProfile]`, `get(name) -> SurveillanceProfile | None`
  - `get_active() -> SurveillanceProfile`, propriété `active_name: str`, `set_active(name) -> bool`
  - `save_profile(profile) -> None`, `delete_profile(name) -> bool`
  - `migrate_alert_fields() -> bool` — importe les champs d'alerte d'un `alerts_config.json` existant vers les profils, une seule fois

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/storage/test_profile_repository.py` :

```python
import json

import pytest

from face_recognition_app.domain.profile import SurveillanceProfile
from face_recognition_app.settings import AppSettings
from face_recognition_app.storage.profile_repository import ProfileRepository


@pytest.fixture
def settings(tmp_path):
    s = AppSettings.create(project_root=tmp_path)
    s.ensure_directories()
    return s


def test_profils_par_defaut_disponibles_sans_fichier(settings):
    repo = ProfileRepository(settings)
    assert {p.name for p in repo.list_all()} == {"present", "absent", "nuit"}


def test_profil_actif_par_defaut(settings):
    assert ProfileRepository(settings).active_name == "present"


def test_changement_de_profil_actif_persiste(settings):
    ProfileRepository(settings).set_active("absent")
    assert ProfileRepository(settings).active_name == "absent"


def test_set_active_sur_profil_inconnu_refuse(settings):
    repo = ProfileRepository(settings)
    assert repo.set_active("inexistant") is False
    assert repo.active_name == "present"


def test_profil_personnalise_persiste(settings):
    repo = ProfileRepository(settings)
    repo.save_profile(SurveillanceProfile(name="vacances", label="Vacances", record_video=True))
    assert ProfileRepository(settings).get("vacances").record_video is True


def test_suppression_d_un_profil_par_defaut_refusee(settings):
    assert ProfileRepository(settings).delete_profile("present") is False


def test_suppression_d_un_profil_personnalise(settings):
    repo = ProfileRepository(settings)
    repo.save_profile(SurveillanceProfile(name="vacances", label="Vacances"))
    assert repo.delete_profile("vacances") is True
    assert repo.get("vacances") is None


def test_suppression_du_profil_actif_bascule_sur_present(settings):
    repo = ProfileRepository(settings)
    repo.save_profile(SurveillanceProfile(name="vacances", label="Vacances"))
    repo.set_active("vacances")
    repo.delete_profile("vacances")
    assert repo.active_name == "present"


def test_fichier_illisible_retombe_sur_les_defauts(settings):
    settings.profiles_file.write_text("{ pas du json")
    repo = ProfileRepository(settings)
    assert repo.active_name == "present"
    assert len(repo.list_all()) == 3


def test_profiles_json_existant_relu(settings):
    settings.profiles_file.write_text(
        json.dumps(
            {
                "active": "nuit",
                "profiles": [
                    {
                        "name": "nuit",
                        "label": "Nuit",
                        "record_video": True,
                        "enabled_camera_uids": ["obsolete"],
                    }
                ],
            }
        )
    )
    repo = ProfileRepository(settings)
    assert repo.active_name == "nuit"
    assert repo.get("nuit").record_video is True


def test_migration_importe_les_champs_d_alerte(settings):
    settings.alerts_file.write_text(
        json.dumps(
            {
                "alert_on_unknown": False,
                "alert_on_known": True,
                "target_persons": ["Bob"],
                "smtp_host": "smtp.example.com",
            }
        )
    )
    repo = ProfileRepository(settings)
    assert repo.migrate_alert_fields() is True

    actif = ProfileRepository(settings).get_active()
    assert actif.alert_on_known is True
    assert actif.target_persons == ["Bob"]

    restant = json.loads(settings.alerts_file.read_text())
    assert "alert_on_known" not in restant
    assert restant["smtp_host"] == "smtp.example.com"


def test_migration_sans_fichier_ne_fait_rien(settings):
    assert ProfileRepository(settings).migrate_alert_fields() is False


def test_migration_idempotente(settings):
    settings.alerts_file.write_text(json.dumps({"alert_on_known": True}))
    repo = ProfileRepository(settings)
    assert repo.migrate_alert_fields() is True
    assert repo.migrate_alert_fields() is False
```

- [ ] **Step 2: Vérifier que les tests échouent**

Run: `.venv/bin/python -m pytest tests/storage/test_profile_repository.py`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implémenter**

`src/face_recognition_app/storage/profile_repository.py` :

```python
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
            name: SurveillanceProfile.from_dict(p.to_dict())
            for name, p in DEFAULT_PROFILES.items()
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
```

- [ ] **Step 4: Vérifier que les tests passent**

Run: `.venv/bin/python -m pytest tests/storage/test_profile_repository.py`
Expected: `13 passed`

- [ ] **Step 5: Transformer l'ancien module en adaptateur**

Remplacer intégralement `src/face_recognition_app/storage/profile_store.py` :

```python
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
```

- [ ] **Step 6: Vérifier la suite et le fichier réel**

Run: `.venv/bin/python -m pytest`
Expected: `91 passed`

Run: `.venv/bin/python -c "
import sys; sys.path.insert(0,'src')
from face_recognition_app.storage.profile_store import ProfileStore
s = ProfileStore()
print('actif:', s.active_name, '| profils:', [p.name for p in s.list_profiles()])
"`
Expected: `actif: present | profils: ['present', 'absent', 'nuit']`

- [ ] **Step 7: Commit**

```bash
git add -A src/face_recognition_app/storage tests/storage
git commit -m "refactor: ProfileRepository et migration des champs d'alerte

Les champs alert_on_unknown, alert_on_known et target_persons vivaient
en double entre SurveillanceProfile et AlertConfig, seule la version
AlertConfig ayant un effet. migrate_alert_fields() les importe dans les
profils et les retire d'alerts_config.json, une seule fois.

Ajoute la validation d'un profil illisible, qui retombe desormais sur
les defauts au lieu de propager l'erreur."
```

---

### Task 10: `storage/camera_repository.py` — persistance des caméras

**Files:**
- Create: `src/face_recognition_app/storage/camera_repository.py`
- Create: `tests/storage/test_camera_repository.py`
- Modify: `src/face_recognition_app/services/camera_source.py` (importe `CameraConfig` depuis le domaine)

**Interfaces:**
- Consumes: `AppSettings` (Task 3), `CameraConfig` (Task 4)
- Produces: `CameraRepository(settings: AppSettings)` avec `load_all() -> list[CameraConfig]` et `save_all(configs: Iterable[CameraConfig]) -> None`

Le repository ne gère que la persistance. Le cycle de vie des sources reste dans `CameraManager`, refactorisé en Phase 2.

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/storage/test_camera_repository.py` :

```python
import json

import pytest

from face_recognition_app.domain.camera import CameraConfig
from face_recognition_app.settings import AppSettings
from face_recognition_app.storage.camera_repository import CameraRepository


@pytest.fixture
def repo(tmp_path):
    settings = AppSettings.create(project_root=tmp_path)
    settings.ensure_directories()
    return CameraRepository(settings)


def test_liste_vide_sans_fichier(repo):
    assert repo.load_all() == []


def test_aller_retour(repo):
    configs = [
        CameraConfig(name="Salon", source_type="webcam", source=0),
        CameraConfig(name="Couloir", source_type="ip", source="http://10.0.0.5:8080/video"),
    ]
    repo.save_all(configs)
    assert [c.name for c in repo.load_all()] == ["Salon", "Couloir"]


def test_fichier_illisible_retourne_liste_vide(repo, tmp_path):
    (tmp_path / "cameras.json").write_text("{ pas du json")
    assert repo.load_all() == []


def test_entree_invalide_ignoree_les_autres_conservees(repo, tmp_path):
    (tmp_path / "cameras.json").write_text(
        json.dumps(
            [
                {"name": "Bonne", "source_type": "webcam", "source": 0},
                {"name": "Mauvaise", "source_type": "satellite", "source": 0},
            ]
        )
    )
    assert [c.name for c in repo.load_all()] == ["Bonne"]


def test_format_existant_relu(repo, tmp_path):
    (tmp_path / "cameras.json").write_text(
        json.dumps(
            [
                {
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
            ]
        )
    )
    configs = repo.load_all()
    assert len(configs) == 1
    assert configs[0].uid == "6dc1bce5"
```

- [ ] **Step 2: Vérifier que les tests échouent**

Run: `.venv/bin/python -m pytest tests/storage/test_camera_repository.py`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implémenter**

`src/face_recognition_app/storage/camera_repository.py` :

```python
"""
camera_repository.py
Persistance des configurations de caméras dans `cameras.json`.

Une entrée invalide est ignorée et journalisée, sans empêcher le chargement des
autres : une caméra mal configurée ne doit pas priver l'utilisateur de tout son
système de surveillance.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable

from ..domain.camera import CameraConfig
from ..settings import AppSettings

logger = logging.getLogger(__name__)


class CameraRepository:
    """Lit et écrit les configurations de caméras."""

    def __init__(self, settings: AppSettings) -> None:
        self._file = settings.cameras_file

    def load_all(self) -> list[CameraConfig]:
        if not self._file.exists():
            return []
        try:
            items = json.loads(self._file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("%s illisible, aucune caméra chargée : %s", self._file.name, exc)
            return []

        configs: list[CameraConfig] = []
        for item in items:
            try:
                configs.append(CameraConfig.from_dict(item))
            except (KeyError, TypeError, ValueError) as exc:
                logger.error("Caméra invalide ignorée (%s) : %s", item.get("name", "?"), exc)
        logger.info("%d caméra(s) chargée(s)", len(configs))
        return configs

    def save_all(self, configs: Iterable[CameraConfig]) -> None:
        payload = [c.to_dict() for c in configs]
        try:
            self._file.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as exc:
            logger.error("Écriture de %s échouée : %s", self._file.name, exc)
```

- [ ] **Step 4: Vérifier que les tests passent**

Run: `.venv/bin/python -m pytest tests/storage/test_camera_repository.py`
Expected: `5 passed`

- [ ] **Step 5: Faire pointer `camera_source` vers le `CameraConfig` du domaine**

Dans `src/face_recognition_app/services/camera_source.py`, supprimer la définition locale de `CameraConfig` (lignes 30 à 76, du décorateur `@dataclass` jusqu'à la fin de `from_dict`) ainsi que les imports devenus inutiles, et importer le modèle du domaine.

Remplacer le bloc d'imports en tête de fichier :

```python
from __future__ import annotations

import logging
import threading
import time
import uuid
from abc import ABC
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)
```

par :

```python
from __future__ import annotations

import logging
import threading
import time
from abc import ABC
from typing import Optional

import cv2
import numpy as np

from ..domain.camera import CameraConfig

__all__ = ["CameraConfig", "CameraSource", "IPCameraSource", "WebcamSource", "create_camera_source"]

logger = logging.getLogger(__name__)
```

`CameraConfig` reste réexporté depuis ce module : `ui/camera_config_dialog.py` et `ui/surveillance_dashboard.py` l'importent de là et continuent de fonctionner sans modification.

- [ ] **Step 6: Vérifier que tout tient debout**

Run: `.venv/bin/ruff check src tests`
Expected: `All checks passed!`

Run: `.venv/bin/python -m pytest`
Expected: `96 passed`

Run: `.venv/bin/python -c "
import sys; sys.path.insert(0,'src')
from face_recognition_app.services.camera_source import CameraConfig, create_camera_source
from face_recognition_app.settings import AppSettings
from face_recognition_app.storage.camera_repository import CameraRepository
configs = CameraRepository(AppSettings.create()).load_all()
print(f'{len(configs)} cameras :', [c.name for c in configs])
print('fabrique ok:', type(create_camera_source(configs[0])).__name__)
"`
Expected: `4 cameras : ['camera couloir', 'webcam', 'cam1', 'escalier']` puis `fabrique ok: IPCameraSource`

- [ ] **Step 7: Commit**

```bash
git add -A src/face_recognition_app tests/storage
git commit -m "refactor: CameraRepository, CameraConfig remonte dans le domaine

Une camera mal configuree est desormais ignoree et journalisee sans
empecher le chargement des autres.

camera_source reexporte CameraConfig pour que l'UI continue de
l'importer depuis le meme endroit."
```

---

### Task 11: Câblage — logging centralisé et composition root

**Files:**
- Create: `src/face_recognition_app/logging_config.py`
- Modify: `src/face_recognition_app/__main__.py`
- Modify: `src/face_recognition_app/ui/interface.py:16-19` (retrait de `basicConfig`)
- Modify: `src/face_recognition_app/ui/video_importer.py:26` (retrait de `basicConfig`)
- Modify: `src/face_recognition_app/storage/config.py` (devient un adaptateur)
- Create: `tests/test_logging_config.py`

**Interfaces:**
- Consumes: `AppSettings` (Task 3), tous les repositories (Tasks 7-10)
- Produces: `configure_logging(settings: AppSettings, verbose: bool = False) -> None` ; `build_context(settings) -> AppContext` (dataclass regroupant `settings`, `encodings`, `events`, `profiles`, `cameras`)

**Pourquoi :** `ui/video_importer.py:26` appelle `logging.basicConfig(filename='app.log')` **à l'import**, ce qui redirige tout le logging de l'application vers un fichier relatif au répertoire courant. `ui/interface.py:16` en appelle un autre vers stdout. Le premier import gagne, silencieusement.

- [ ] **Step 1: Écrire le test qui échoue**

Créer `tests/test_logging_config.py` :

```python
import logging

from face_recognition_app.logging_config import configure_logging
from face_recognition_app.settings import AppSettings


def test_configure_logging_ecrit_dans_le_fichier(tmp_path):
    settings = AppSettings.create(project_root=tmp_path)
    settings.ensure_directories()
    configure_logging(settings)
    logging.getLogger("test").warning("message de controle")
    logging.shutdown()

    contenu = (tmp_path / "app.log").read_text(encoding="utf-8")
    assert "message de controle" in contenu


def test_configure_logging_est_idempotent(tmp_path):
    settings = AppSettings.create(project_root=tmp_path)
    settings.ensure_directories()
    configure_logging(settings)
    avant = len(logging.getLogger().handlers)
    configure_logging(settings)
    assert len(logging.getLogger().handlers) == avant


def test_mode_verbeux_abaisse_le_niveau(tmp_path):
    settings = AppSettings.create(project_root=tmp_path)
    settings.ensure_directories()
    configure_logging(settings, verbose=True)
    assert logging.getLogger().level == logging.DEBUG
```

- [ ] **Step 2: Vérifier que les tests échouent**

Run: `.venv/bin/python -m pytest tests/test_logging_config.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'face_recognition_app.logging_config'`

- [ ] **Step 3: Implémenter la configuration du logging**

`src/face_recognition_app/logging_config.py` :

```python
"""
logging_config.py
Configuration unique du logging, appliquée depuis le point d'entrée.

Aucun module ne doit appeler `logging.basicConfig` : deux modules le faisaient à
l'import, avec des destinations contradictoires, et le premier importé gagnait
silencieusement.

Sortie : console (INFO) et fichier tournant `app.log` à la racine du projet.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from .settings import AppSettings

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s — %(message)s"
_MAX_BYTES = 2 * 1024 * 1024
_BACKUPS = 3

_configured = False


def configure_logging(settings: AppSettings, verbose: bool = False) -> None:
    """Installe les handlers racine. Idempotent."""
    global _configured
    if _configured:
        return

    niveau = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter(_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.setLevel(niveau)

    fichier = RotatingFileHandler(
        settings.project_root / "app.log",
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUPS,
        encoding="utf-8",
    )
    fichier.setFormatter(formatter)
    fichier.setLevel(logging.DEBUG)

    racine = logging.getLogger()
    racine.setLevel(niveau)
    racine.handlers.clear()
    racine.addHandler(console)
    racine.addHandler(fichier)

    # Flask est bavard sur chaque requête ; on ne garde que ses erreurs.
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    _configured = True
```

- [ ] **Step 4: Vérifier que les tests passent**

Run: `.venv/bin/python -m pytest tests/test_logging_config.py`
Expected: `3 passed`

- [ ] **Step 5: Retirer les `basicConfig` des modules UI**

Dans `src/face_recognition_app/ui/interface.py`, supprimer :

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
```

en conservant `logger = logging.getLogger(__name__)`.

Dans `src/face_recognition_app/ui/video_importer.py`, supprimer :

```python
logging.basicConfig(filename='app.log', level=logging.INFO)
```

et ajouter juste après les imports :

```python
logger = logging.getLogger(__name__)
```

puis remplacer dans `log_error` l'appel `logging.error(message)` par `logger.error(message)`.

- [ ] **Step 6: Transformer `storage/config.py` en adaptateur**

Remplacer intégralement `src/face_recognition_app/storage/config.py` :

```python
"""
config.py
ADAPTATEUR TEMPORAIRE — conserve les constantes utilisées par services/ et ui/.

Les valeurs proviennent désormais d'AppSettings. Ce module sera supprimé en
Phase 3, quand tous les appelants auront reçu la configuration par injection.

Les trois seuils historiques sont ramenés à deux :
    FACE_RECOGNITION_THRESHOLD → profil.recognition_threshold
    DUPLICATE_TOLERANCE, VIDEO_FACE_TOLERANCE → settings.duplicate_tolerance
"""

from __future__ import annotations

from ..domain.profile import DEFAULT_PROFILES
from ..settings import AppSettings

_settings = AppSettings.create()
_settings.ensure_directories()

PROJECT_ROOT = _settings.project_root
ENCODED_DIR = str(_settings.encodings_dir)
META_FILE = str(_settings.encodings_dir / "metadata.json")
CAMERAS_FILE = _settings.cameras_file

DUPLICATE_TOLERANCE = _settings.duplicate_tolerance
VIDEO_FACE_TOLERANCE = _settings.duplicate_tolerance
FACE_RECOGNITION_THRESHOLD = DEFAULT_PROFILES["present"].recognition_threshold
```

- [ ] **Step 7: Câbler le point d'entrée**

Dans `src/face_recognition_app/__main__.py`, remplacer intégralement le fichier :

```python
"""
__main__.py
Point d'entrée de l'application — composition root.

Construit la configuration, installe le logging, instancie les repositories,
applique les migrations de démarrage, puis lance le tableau de bord.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import ttkbootstrap as ttk

from .logging_config import configure_logging
from .settings import AppSettings
from .storage.camera_repository import CameraRepository
from .storage.encodings_repository import EncodingsRepository
from .storage.event_repository import EventRepository
from .storage.profile_repository import ProfileRepository

logger = logging.getLogger(__name__)


@dataclass
class AppContext:
    """Composants partagés, construits une fois et passés explicitement."""

    settings: AppSettings
    encodings: EncodingsRepository
    events: EventRepository
    profiles: ProfileRepository
    cameras: CameraRepository


def _verifier_opencv() -> None:
    """
    Signale l'installation simultanée d'opencv-python et opencv-python-headless.

    La variante headless l'emporte à l'import et casse silencieusement toute
    fonction d'affichage OpenCV (bug ⑫).
    """
    try:
        from importlib.metadata import distributions
    except ImportError:  # pragma: no cover — stdlib depuis 3.8
        return
    installes = {d.metadata["Name"] for d in distributions() if d.metadata["Name"]}
    if {"opencv-python", "opencv-python-headless"} <= installes:
        logger.warning(
            "opencv-python et opencv-python-headless sont installés ensemble : "
            "l'affichage OpenCV sera cassé. Exécutez « pip uninstall opencv-python-headless »."
        )


def build_context(settings: AppSettings) -> AppContext:
    """Instancie les repositories et applique les migrations de démarrage."""
    settings.ensure_directories()

    profiles = ProfileRepository(settings)
    if profiles.migrate_alert_fields():
        logger.info("Migration des champs d'alerte appliquée")

    events = EventRepository(settings)
    events.start()
    supprimes = events.purge_expired()
    if supprimes:
        events.vacuum()

    context = AppContext(
        settings=settings,
        encodings=EncodingsRepository(settings),
        events=events,
        profiles=profiles,
        cameras=CameraRepository(settings),
    )

    # L'adaptateur d'encodages partage le repository de l'application.
    from .storage import encodings_store

    encodings_store.set_repository(context.encodings)

    return context


def main() -> None:
    settings = AppSettings.create()
    configure_logging(settings)
    logger.info("Démarrage — racine projet : %s", settings.project_root)

    _verifier_opencv()
    context = build_context(settings)

    root = ttk.Window(themename="solar")
    root.withdraw()

    from .ui.surveillance_dashboard import SurveillanceDashboard

    SurveillanceDashboard(root)

    try:
        root.mainloop()
    finally:
        context.events.stop()
        logger.info("Arrêt terminé")


if __name__ == "__main__":
    main()
```

Note : le menu à six boutons disparaît, conformément à la spec §4.5 — le dashboard devient la fenêtre de démarrage. Les modules v1 restent présents sur disque jusqu'à la Phase 3, mais ne sont plus atteignables depuis le lanceur.

- [ ] **Step 8: Vérifier la suite et le démarrage réel**

Run: `.venv/bin/ruff check src tests && .venv/bin/python -m pytest`
Expected: `All checks passed!` puis `99 passed`

Run: `.venv/bin/python -c "
import sys; sys.path.insert(0,'src')
from face_recognition_app.__main__ import build_context
from face_recognition_app.settings import AppSettings
from face_recognition_app.logging_config import configure_logging
s = AppSettings.create()
configure_logging(s)
ctx = build_context(s)
print('encodages:', sorted(ctx.encodings.load_map()))
print('cameras:', [c.name for c in ctx.cameras.load_all()])
print('profil actif:', ctx.profiles.active_name)
print('evenements:', ctx.events.count())
ctx.events.stop()
"`
Expected: les quatre lignes affichent les données réelles sans exception.

- [ ] **Step 9: Vérifier que l'application démarre**

Run: `timeout 10 .venv/bin/python main.py; echo "code de sortie: $?"`
Expected: la fenêtre du dashboard s'ouvre puis `timeout` l'interrompt — `code de sortie: 124`. Tout autre code signale une exception au démarrage, à corriger avant de commiter.

- [ ] **Step 10: Commit**

```bash
git add -A src/face_recognition_app tests
git commit -m "refactor: composition root et logging centralise

__main__ construit AppSettings, installe le logging, instancie les
repositories, applique la migration des alertes et la purge des
evenements expires, puis lance le dashboard.

Retire les deux logging.basicConfig appeles a l'import, dont l'un
redirigeait tout le logging vers app.log dans le repertoire courant :
le premier module importe gagnait silencieusement.

storage/config.py devient un adaptateur ; les trois seuils historiques
sont ramenes a deux (duplicate_tolerance et recognition_threshold)."
```

---

### Task 12: Intégration continue

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: `pyproject.toml` (Task 1), tous les tests
- Produces: une CI qui exécute ruff et pytest sans compiler dlib

**Pourquoi c'est possible :** aucun test n'importe `face_recognition` ni `dlib` — la comparaison de visages vit dans `domain/matching.py` en numpy pur (Task 2). La CI n'installe donc que numpy, OpenCV headless, Flask et Pillow, ce qui la fait tourner en moins de deux minutes au lieu des dix minutes de compilation de dlib.

- [ ] **Step 1: Vérifier localement qu'aucun test ne dépend de dlib**

Run: `.venv/bin/python -c "
import subprocess, sys
code = '''
import sys, pytest
sys.exit(pytest.main(['-q', 'tests']))
'''
r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
print(r.stdout[-400:])
assert 'dlib' not in r.stdout, 'un test importe dlib'
print('aucun test ne depend de dlib')
"`
Expected: `aucun test ne depend de dlib`

- [ ] **Step 2: Créer le workflow**

`.github/workflows/ci.yml` :

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  qualite:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip

      - name: Installer les dépendances de test
        # dlib et face_recognition ne sont pas installés : la comparaison de
        # visages vit dans domain/matching.py en numpy pur, et le moteur de
        # reconnaissance est mocké à la frontière face_matcher.
        run: |
          python -m pip install --upgrade pip
          pip install \
            numpy==2.4.4 \
            opencv-python-headless==4.13.0.92 \
            Pillow==12.2.0 \
            flask==3.1.3 \
            requests==2.33.1 \
            pytest==9.0.2 \
            ruff==0.9.6 \
            mypy==1.15.0

      - name: Lint
        run: ruff check src tests

      - name: Format
        run: ruff format --check src tests

      - name: Types (domaine et configuration)
        run: mypy

      - name: Tests
        run: pytest
```

- [ ] **Step 3: Vérifier que les commandes de la CI passent localement**

Run: `.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests && .venv/bin/python -m pytest`
Expected: `All checks passed!`, aucun fichier à reformater, `99 passed`

- [ ] **Step 4: Vérifier mypy**

```bash
.venv/bin/pip install mypy==1.15.0
```

Run: `.venv/bin/mypy`
Expected: `Success: no issues found`

Si mypy signale des erreurs sur `domain/` ou `settings.py`, les corriger — ces deux couches doivent être typées proprement, c'est le contrat posé dans `pyproject.toml`.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "chore: CI GitHub Actions, ruff + mypy + pytest

La CI n'installe ni dlib ni face_recognition : la comparaison de visages
vit dans domain/matching.py en numpy pur et le moteur sera mocke a la
frontiere face_matcher. Environ deux minutes au lieu de dix."
```

---

## Vérification de fin de phase

- [ ] **Suite complète verte**

Run: `.venv/bin/python -m pytest`
Expected: `99 passed`

- [ ] **Qualité**

Run: `.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests && .venv/bin/mypy`
Expected: aucun problème

- [ ] **Données intactes**

Run: `.venv/bin/python -c "
import sys; sys.path.insert(0,'src')
from face_recognition_app.settings import AppSettings
from face_recognition_app.storage.encodings_repository import EncodingsRepository
from face_recognition_app.storage.camera_repository import CameraRepository
from face_recognition_app.storage.event_repository import EventRepository
s = AppSettings.create()
assert sorted(EncodingsRepository(s).load_map()) == ['ravaka', 'vic', 'victorien']
assert len(CameraRepository(s).load_all()) == 4
e = EventRepository(s); assert e.count() > 0; e.stop()
print('donnees existantes intactes')
"`
Expected: `donnees existantes intactes`

- [ ] **Application fonctionnelle**

Run: `timeout 10 .venv/bin/python main.py; echo "code: $?"`
Expected: `code: 124` (la fenêtre s'est ouverte et le timeout l'a interrompue)

---

## Écart assumé par rapport à la spec

La spec (§10, lot 0) prévoyait un « filet de caractérisation » écrit sur les services
existants **avant** toute modification. Ce plan le remplace par des tests écrits en TDD
au moment où chaque module est créé.

La raison : les services (`camera_manager`, `motion_detector`, `alert_manager`,
`surveillance_engine`) sont refondus en Phase 2, pas en Phase 1. Écrire maintenant des
tests contre leur API actuelle reviendrait à les réécrire deux semaines plus tard. Les
couches touchées par la Phase 1 — domaine et stockage — sont bien couvertes avant
d'être branchées, ce qui préserve l'intention du filet là où elle protège réellement.

Le filet de caractérisation des services est donc déplacé en tête de Phase 2, écrit
juste avant la refonte de chaque service.

## Ce que la Phase 1 ne fait pas

Ces points relèvent des Phases 2 et 3, et sont volontairement laissés en l'état :

- La boucle de reconnexion caméra reste défaillante (bug ③) — Phase 2, lot 3.
- Le moteur applique toujours partiellement le profil (bugs ⑤⑥⑦) — Phase 2, lot 4.
- L'API reste sans authentification (bug ④ et sécurité) — Phase 2, lot 6.
- Les modules v1 restent sur disque, désormais inatteignables depuis le lanceur — Phase 3, lot 7.
- Les bugs ① ② ⑧ ⑨ ⑩ ⑪ vivent dans du code UI traité en Phase 3.
- Les adaptateurs `storage/config.py`, `encodings_store.py`, `event_store.py`, `profile_store.py` sont supprimés en Phase 3, une fois tous leurs appelants convertis à l'injection.
