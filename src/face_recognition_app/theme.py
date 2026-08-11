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
# Les maquettes font du vert l'accent de marque — c'est lui qui donne le
# caractère « poste de contrôle » : démarrage, onglet actif, pastille d'API.
# Il porte donc deux rôles à la fois, marque et état sain, comme dans la
# maquette. Le bleu recule sur la sélection et les liens.
BRAND_ACCENT = "#22C55E"  # vert signal — action principale, onglet actif
BRAND_PRIMARY = "#2563EB"  # bleu électrique — sélection, liens
ACCENT_AI = "#06B6D4"  # cyan — indicateurs d'analyse

# États
STATE_OK = "#22C55E"  # vert — accès autorisé, flux sain
STATE_WARN = "#F59E0B"  # orange — attention, reconnexion
STATE_DANGER = "#EF4444"  # rouge — intrusion, panne
AI_TRACKING = "#8B5CF6"  # violet — personne suivie

# Bordures — la profondeur passe par des traits de 1 px, jamais par des ombres.
# Deux niveaux : `BORDER` cerne les panneaux, `BORDER_SUBTLE` sépare les lignes
# d'une même liste. Un seul niveau ne suffit pas — une bordure de la couleur des
# cartes serait invisible sur une carte.
# En-tête de panneau en anomalie : rouge très désaturé, lisible sans crier.
HEADER_ALERT = "#2A1518"
BORDER = "#334155"
BORDER_SUBTLE = "#1E293B"
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
        "BRAND_ACCENT": BRAND_ACCENT,
        "HEADER_ALERT": HEADER_ALERT,
        "BORDER": BORDER,
        "BORDER_SUBTLE": BORDER_SUBTLE,
        "SURFACE_HOVER": SURFACE_HOVER,
    }


# ── Polices ───────────────────────────────────────────────────────────────────

# Les maquettes demandent Inter et JetBrains Mono. Aucune des deux n'est
# garantie sur un poste Linux — et « Helvetica » ne l'est pas davantage, Tk la
# substituant alors en silence. On résout donc la première famille réellement
# installée, par ordre de préférence.
_PREF_SANS = ("Inter", "Noto Sans", "DejaVu Sans", "Liberation Sans", "Ubuntu", "TkDefaultFont")
_PREF_MONO = (
    "JetBrains Mono",
    "DejaVu Sans Mono",
    "Liberation Mono",
    "Ubuntu Mono",
    "TkFixedFont",
)


def _premiere_disponible(preferences: tuple[str, ...], defaut: str) -> str:
    """
    Première famille de polices installée parmi les préférences.

    Retourne `defaut` si Tk n'est pas joignable — le module doit rester
    importable sans écran.
    """
    try:
        from tkinter import font as tkfont

        installees = set(tkfont.families())
    except Exception:
        return defaut
    return next((f for f in preferences if f in installees), defaut)


_sans: str | None = None
_mono: str | None = None


def font_sans() -> str:
    """Famille sans-serif retenue, résolue une seule fois."""
    global _sans
    if _sans is None:
        _sans = _premiere_disponible(_PREF_SANS, "TkDefaultFont")
    return _sans


def font_mono() -> str:
    """Famille monospace retenue, résolue une seule fois."""
    global _mono
    if _mono is None:
        _mono = _premiere_disponible(_PREF_MONO, "TkFixedFont")
    return _mono


# Échelle dense, calquée sur les maquettes : peu de tailles, beaucoup de
# contraste par la graisse et la casse.
def FONT_TITLE() -> tuple:  # noqa: N802 — jetons de thème, pas des classes
    return (font_sans(), 17, "bold")


def FONT_NAV() -> tuple:  # noqa: N802
    return (font_sans(), 10, "bold")


def FONT_HEADING() -> tuple:  # noqa: N802
    return (font_sans(), 10, "bold")


def FONT_BODY() -> tuple:  # noqa: N802
    return (font_sans(), 10)


def FONT_SMALL() -> tuple:  # noqa: N802
    return (font_sans(), 9)


def FONT_BADGE() -> tuple:  # noqa: N802
    return (font_mono(), 8, "bold")


def FONT_MONO() -> tuple:  # noqa: N802
    return (font_mono(), 9)


def FONT_DATA() -> tuple:  # noqa: N802
    return (font_mono(), 10)


# ── Espacements ───────────────────────────────────────────────────────────────

PAD_XS = 4
PAD_S = 8
PAD_M = 12
PAD_L = 16
PAD_XL = 24
BORDER_W = 1


# ── Thème ttkbootstrap ────────────────────────────────────────────────────────

THEME_NAME = "surveillance"

# Projection de la palette sur le modèle de couleurs de ttkbootstrap.
# Les seize emplacements sont obligatoires ; `Colors` les exige tous.
TTK_COLORS: dict[str, str] = {
    "primary": BRAND_ACCENT,
    "secondary": TEXT_SECONDARY,
    "success": BRAND_ACCENT,
    "info": BRAND_PRIMARY,
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
