"""
theme.py
Palette et jetons visuels de l'application — source unique de vérité.

L'interface comptait 88 couleurs codées en dur pour 26 valeurs distinctes,
héritées de plusieurs générations de code. Elles sont remplacées par les jetons
nommés ci-dessous.

Le module remplit aussi deux rôles moins évidents :

  - **Cohérence ttk / tk.** `ttkbootstrap` colore lui-même les widgets `ttk.*`
    depuis son propre modèle de couleurs. `TTK_COLORS` y projette la palette, si
    bien qu'un `ttk.Button` et un `tk.Frame` partagent le même fond. Sans cela,
    on obtenait des panneaux bleu nuit garnis de boutons orange « solar ».
  - **Annotations OpenCV.** Les rectangles dessinés sur les visages viennent de
    la même palette, via `to_bgr()` — OpenCV attend du BGR, pas du RGB.

Le module vit à la racine du paquet et non dans `ui/` : les services y
puisent aussi, pour les annotations OpenCV, et un service ne doit jamais
importer la couche de présentation.

Aucune dépendance à Tk au niveau module : `apply_theme` importe ttkbootstrap
paresseusement, ce qui rend le reste importable et testable sans écran.
"""

from __future__ import annotations

from .domain.connection import ConnectionState

# ── Palette ───────────────────────────────────────────────────────────────────

# Fonds, du plus profond au plus clair
BG_BASE = "#0B1120"  # fond principal, noir bleuté
BG_SURFACE = "#111827"  # fond secondaire, bleu nuit — barres, en-têtes
BG_CARD = "#1E293B"  # cartes et panneaux, slate foncé

# Texte
TEXT_PRIMARY = "#F8FAFC"  # blanc cassé
TEXT_SECONDARY = "#94A3B8"  # gris bleuté — libellés, métadonnées

# Marque et accents
BRAND_PRIMARY = "#2563EB"  # bleu électrique — action principale
ACCENT_AI = "#06B6D4"  # cyan — indicateurs d'analyse

# États
STATE_OK = "#22C55E"  # vert — accès autorisé, flux sain
STATE_WARN = "#F59E0B"  # orange — attention, reconnexion
STATE_DANGER = "#EF4444"  # rouge — intrusion, panne
AI_TRACKING = "#8B5CF6"  # violet — personne suivie

# Dérivés — bordures et survol, calés sur les fonds pour rester cohérents
BORDER = BG_CARD
SURFACE_HOVER = "#243449"


def tokens() -> dict[str, str]:
    """Tous les jetons de couleur, pour vérification et introspection."""
    return {
        "BG_BASE": BG_BASE,
        "BG_SURFACE": BG_SURFACE,
        "BG_CARD": BG_CARD,
        "TEXT_PRIMARY": TEXT_PRIMARY,
        "TEXT_SECONDARY": TEXT_SECONDARY,
        "BRAND_PRIMARY": BRAND_PRIMARY,
        "ACCENT_AI": ACCENT_AI,
        "STATE_OK": STATE_OK,
        "STATE_WARN": STATE_WARN,
        "STATE_DANGER": STATE_DANGER,
        "AI_TRACKING": AI_TRACKING,
        "BORDER": BORDER,
        "SURFACE_HOVER": SURFACE_HOVER,
    }


# ── Polices ───────────────────────────────────────────────────────────────────

_FAMILLE = "Helvetica"

FONT_TITLE = (_FAMILLE, 15, "bold")
FONT_HEADING = (_FAMILLE, 11, "bold")
FONT_BODY = (_FAMILLE, 10)
FONT_SMALL = (_FAMILLE, 9)
FONT_BADGE = (_FAMILLE, 8, "bold")
FONT_MONO = ("Courier", 9)


# ── Espacements ───────────────────────────────────────────────────────────────

PAD_XS = 2
PAD_S = 4
PAD_M = 8
PAD_L = 12
PAD_XL = 16


# ── Thème ttkbootstrap ────────────────────────────────────────────────────────

THEME_NAME = "surveillance"

# Projection de la palette sur le modèle de couleurs de ttkbootstrap.
# Les seize emplacements sont obligatoires ; `Colors` les exige tous.
TTK_COLORS: dict[str, str] = {
    "primary": BRAND_PRIMARY,
    "secondary": TEXT_SECONDARY,
    "success": STATE_OK,
    "info": ACCENT_AI,
    "warning": STATE_WARN,
    "danger": STATE_DANGER,
    "light": TEXT_PRIMARY,
    "dark": BG_BASE,
    "bg": BG_BASE,
    "fg": TEXT_PRIMARY,
    "selectbg": BRAND_PRIMARY,
    "selectfg": TEXT_PRIMARY,
    "border": BORDER,
    "inputfg": TEXT_PRIMARY,
    "inputbg": BG_SURFACE,
    "active": SURFACE_HOVER,
}


def apply_theme(window) -> None:
    """
    Enregistre le thème et l'applique à la fenêtre ttkbootstrap donnée.

    `Style.register_theme` est une méthode d'instance : il faut donc une fenêtre
    déjà construite. On la crée avec un thème sombre standard, puis on bascule.
    """
    from ttkbootstrap.style import ThemeDefinition

    style = window.style
    if THEME_NAME not in style.theme_names():
        style.register_theme(ThemeDefinition(THEME_NAME, dict(TTK_COLORS), themetype="dark"))
    style.theme_use(THEME_NAME)


# ── Conversion pour OpenCV ────────────────────────────────────────────────────


def to_bgr(couleur: str) -> tuple[int, int, int]:
    """
    Convertit `#RRGGBB` en triplet BGR, l'ordre attendu par OpenCV.

    Raises:
        ValueError: si la chaîne n'est pas un hexadécimal de six chiffres.
    """
    brut = couleur.lstrip("#")
    if len(brut) != 6:
        raise ValueError(f"Couleur hexadécimale à six chiffres attendue, reçu {couleur!r}")
    try:
        r, g, b = (int(brut[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError as exc:
        raise ValueError(f"Couleur hexadécimale invalide : {couleur!r}") from exc
    return (b, g, r)


def face_box_bgr(is_known: bool, is_target: bool) -> tuple[int, int, int]:
    """
    Couleur du cadre dessiné autour d'un visage détecté.

    La cible prime sur le reste : qu'une personne suivie soit connue est
    secondaire, l'information utile est qu'elle est suivie.
    """
    if is_target:
        return to_bgr(AI_TRACKING)
    return to_bgr(STATE_OK if is_known else STATE_DANGER)


# ── Badges d'état ─────────────────────────────────────────────────────────────


def connection_badge(
    state: ConnectionState | None,
    retry_in: float = 0.0,
) -> tuple[str, str]:
    """
    Libellé et couleur du badge d'état d'une caméra.

    Returns:
        (libellé affiché, couleur du badge)
    """
    if state is ConnectionState.CONNECTED:
        return "EN LIGNE", STATE_OK
    if state is ConnectionState.CONNECTING:
        return "CONNEXION…", STATE_WARN
    if state is ConnectionState.DISCONNECTED and retry_in >= 1.0:
        return f"RECONNEXION {int(retry_in)} s", STATE_WARN
    return "HORS LIGNE", STATE_DANGER
