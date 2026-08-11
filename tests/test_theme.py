import re

import pytest

from face_recognition_app import theme
from face_recognition_app.domain.connection import ConnectionState

HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")


# ── Jetons ────────────────────────────────────────────────────────────────────


def test_tous_les_jetons_sont_des_hex_complets():
    """Pas de raccourci #fff : OpenCV et Tk n'en veulent pas de la meme facon."""
    for nom, valeur in theme.tokens().items():
        assert HEX.match(valeur), f"{nom} = {valeur!r}"


def test_palette_conforme_a_la_charte():
    assert theme.BG_BASE == "#0B1120"
    assert theme.BG_SURFACE == "#111827"
    assert theme.BG_CARD == "#1E293B"
    assert theme.TEXT_PRIMARY == "#F8FAFC"
    assert theme.TEXT_SECONDARY == "#94A3B8"
    assert theme.BRAND_PRIMARY == "#2563EB"
    assert theme.ACCENT_AI == "#06B6D4"
    assert theme.STATE_OK == "#22C55E"
    assert theme.STATE_WARN == "#F59E0B"
    assert theme.STATE_DANGER == "#EF4444"
    assert theme.AI_TRACKING == "#8B5CF6"


def test_aucun_jeton_duplique_sur_un_usage_distinct():
    """Deux roles differents ne doivent pas partager la meme teinte."""
    roles = (theme.BG_BASE, theme.BG_SURFACE, theme.BG_CARD)
    assert len(set(roles)) == len(roles)


# ── Conversion pour OpenCV ────────────────────────────────────────────────────


def test_to_bgr_inverse_les_canaux():
    assert theme.to_bgr("#FF0000") == (0, 0, 255)
    assert theme.to_bgr("#00FF00") == (0, 255, 0)
    assert theme.to_bgr("#0000FF") == (255, 0, 0)


def test_to_bgr_sur_un_jeton_de_la_palette():
    # #22C55E → R=0x22, G=0xC5, B=0x5E → BGR (0x5E, 0xC5, 0x22)
    assert theme.to_bgr(theme.STATE_OK) == (0x5E, 0xC5, 0x22)


def test_to_bgr_tolere_l_absence_de_diese():
    assert theme.to_bgr("22C55E") == theme.to_bgr("#22C55E")


def test_to_bgr_rejette_une_valeur_invalide():
    with pytest.raises(ValueError, match="hexad"):
        theme.to_bgr("#GGGGGG")


# ── Mapping ttkbootstrap ──────────────────────────────────────────────────────


def test_le_mapping_couvre_tous_les_emplacements_attendus():
    import inspect

    from ttkbootstrap.style import Colors

    attendus = set(list(inspect.signature(Colors.__init__).parameters)[1:])
    assert set(theme.TTK_COLORS) == attendus


def test_le_mapping_utilise_la_palette():
    """Le vert est l'accent de marque, comme dans les maquettes."""
    assert theme.TTK_COLORS["primary"] == theme.BRAND_ACCENT
    assert theme.TTK_COLORS["success"] == theme.BRAND_ACCENT
    assert theme.TTK_COLORS["bg"] == theme.BG_BASE
    assert theme.TTK_COLORS["fg"] == theme.TEXT_PRIMARY
    assert theme.TTK_COLORS["danger"] == theme.STATE_DANGER


def test_accent_de_marque_est_le_vert():
    assert theme.BRAND_ACCENT == "#22C55E"


def test_deux_niveaux_de_bordure_distincts():
    """Une bordure de la couleur des cartes serait invisible sur une carte."""
    assert theme.BORDER != theme.BORDER_SUBTLE
    assert theme.BORDER_SUBTLE == theme.BG_CARD


# ── Badge d'état de connexion ─────────────────────────────────────────────────


def test_badge_connecte():
    libelle, couleur = theme.connection_badge(ConnectionState.CONNECTED, 0.0)
    assert libelle == "EN LIGNE"
    assert couleur == theme.STATE_OK


def test_badge_en_cours_de_connexion():
    libelle, couleur = theme.connection_badge(ConnectionState.CONNECTING, 0.0)
    assert libelle == "CONNEXION…"
    assert couleur == theme.STATE_WARN


def test_badge_deconnecte_affiche_le_delai_restant():
    libelle, couleur = theme.connection_badge(ConnectionState.DISCONNECTED, 8.0)
    assert libelle == "RECONNEXION 8 s"
    assert couleur == theme.STATE_WARN


def test_badge_deconnecte_sans_delai_est_une_erreur():
    libelle, couleur = theme.connection_badge(ConnectionState.DISCONNECTED, 0.0)
    assert libelle == "HORS LIGNE"
    assert couleur == theme.STATE_DANGER


def test_badge_arrondit_le_delai():
    libelle, _ = theme.connection_badge(ConnectionState.DISCONNECTED, 7.4)
    assert libelle == "RECONNEXION 7 s"


def test_badge_sans_etat_est_hors_ligne():
    libelle, couleur = theme.connection_badge(None, 0.0)
    assert libelle == "HORS LIGNE"
    assert couleur == theme.STATE_DANGER


# ── Couleur d'annotation d'un visage ──────────────────────────────────────────


def test_visage_connu_en_vert():
    assert theme.face_box_bgr(is_known=True, is_target=False) == theme.to_bgr(theme.STATE_OK)


def test_visage_inconnu_en_rouge():
    assert theme.face_box_bgr(is_known=False, is_target=False) == theme.to_bgr(theme.STATE_DANGER)


def test_personne_ciblee_en_violet():
    """La cible prime sur le fait d'etre connue : c'est l'information utile."""
    assert theme.face_box_bgr(is_known=True, is_target=True) == theme.to_bgr(theme.AI_TRACKING)


# ── Polices ───────────────────────────────────────────────────────────────────


def test_polices_definies():
    for police in (
        theme.FONT_TITLE,
        theme.FONT_NAV,
        theme.FONT_HEADING,
        theme.FONT_BODY,
        theme.FONT_SMALL,
        theme.FONT_BADGE,
        theme.FONT_MONO,
        theme.FONT_DATA,
    ):
        famille, taille = police()[0], police()[1]
        assert isinstance(famille, str) and famille
        assert isinstance(taille, int) and taille > 0


def test_familles_resolues_une_seule_fois():
    assert theme.font_sans() is theme.font_sans()
    assert theme.font_mono() is theme.font_mono()


def test_resolution_prend_la_premiere_installee(monkeypatch):
    monkeypatch.setattr(theme, "_sans", None)
    monkeypatch.setattr(theme, "_premiere_disponible", lambda prefs, defaut: "Noto Sans")
    assert theme.font_sans() == "Noto Sans"
    monkeypatch.setattr(theme, "_sans", None)


def test_defaut_si_aucune_police_disponible():
    assert theme._premiere_disponible(("PoliceInexistante42",), "TkDefaultFont") == "TkDefaultFont"
