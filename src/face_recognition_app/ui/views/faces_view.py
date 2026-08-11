"""
faces_view.py
Vue « Base de visages » — personnes enregistrées.

Remplace un détour qui coûtait cher : le tableau de bord instanciait une
`FaceRecognitionApp` complète — fenêtre de 80 % de l'écran, canvas, caméra —
uniquement pour ouvrir sa boîte de gestion des encodages, la fenêtre parasite
restant visible derrière.

La vue lit le `EncodingsRepository` de l'application et propose l'enrôlement
depuis des images.
"""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

from ... import theme
from ...storage.encodings_repository import EncodingsRepository
from ..widgets import Canvas, Frame, Label
from .base import AppServices, View, entete_panneau

logger = logging.getLogger(__name__)

VIGNETTE = 96
COLONNES = 5


class FacesView(View):
    """Grille des visages enregistrés."""

    TITRE = "BASE DE VISAGES"

    def __init__(self, parent, services: AppServices) -> None:
        super().__init__(parent, services)
        self._repo = EncodingsRepository(services.settings)
        self._photos: list[ImageTk.PhotoImage] = []
        self._selection: str | None = None

    # ── Construction ──────────────────────────────────────────────────────────

    def construire(self) -> None:
        entete = entete_panneau(self, "BASE DE VISAGES")
        self._total_var = tk.StringVar(value="")
        Label(
            entete,
            textvariable=self._total_var,
            bg=theme.BG_SURFACE,
            fg=theme.TEXT_SECONDARY,
            font=theme.FONT_BADGE(),
        ).pack(side=tk.RIGHT)

        barre = Frame(self, bg=theme.BG_BASE, padx=theme.PAD_M, pady=theme.PAD_S)
        barre.pack(fill=tk.X)
        ttk.Button(
            barre,
            text="＋ ENREGISTRER UN VISAGE",
            command=self._enroler,
            bootstyle="success",
        ).pack(side=tk.LEFT)
        ttk.Button(
            barre, text="SUPPRIMER", command=self._supprimer, bootstyle="danger-outline"
        ).pack(side=tk.LEFT, padx=theme.PAD_S)
        ttk.Button(
            barre, text="ACTUALISER", command=self.recharger, bootstyle="secondary-outline"
        ).pack(side=tk.LEFT)

        self._selection_var = tk.StringVar(value="Aucune sélection")
        Label(
            barre,
            textvariable=self._selection_var,
            bg=theme.BG_BASE,
            fg=theme.TEXT_SECONDARY,
            font=theme.FONT_BADGE(),
        ).pack(side=tk.RIGHT)

        # Grille défilante
        cadre = Frame(self, bg=theme.BG_BASE)
        cadre.pack(fill=tk.BOTH, expand=True)
        self._canvas = Canvas(cadre, bg=theme.BG_BASE, highlightthickness=0)
        defilement = ttk.Scrollbar(cadre, orient=tk.VERTICAL, command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=defilement.set)
        defilement.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(fill=tk.BOTH, expand=True)

        self._grille = Frame(self._canvas, bg=theme.BG_BASE)
        self._canvas.create_window((0, 0), window=self._grille, anchor=tk.NW)
        self._grille.bind(
            "<Configure>",
            lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")),
        )

        self.recharger()

    def on_show(self) -> None:
        # Un enrôlement a pu avoir lieu depuis l'import d'images.
        if self._construite:
            self.recharger()

    # ── Contenu ───────────────────────────────────────────────────────────────

    def recharger(self) -> None:
        """Relit la base et redessine la grille."""
        for w in self._grille.winfo_children():
            w.destroy()
        self._photos.clear()
        self._selection = None
        self._selection_var.set("Aucune sélection")

        entrees = self._repo.load_all()
        # Un même nom peut avoir plusieurs encodages ; on n'affiche qu'une carte.
        par_nom: dict[str, int] = {}
        for e in entrees:
            par_nom[e.name] = par_nom.get(e.name, 0) + 1

        self._total_var.set(f"{len(par_nom)} PERSONNE(S) · {len(entrees)} ENCODAGE(S)")

        if not par_nom:
            Label(
                self._grille,
                text="AUCUN VISAGE ENREGISTRÉ\n\nUtilisez « ＋ ENREGISTRER UN VISAGE »",
                bg=theme.BG_BASE,
                fg=theme.TEXT_SECONDARY,
                font=theme.FONT_BADGE(),
                justify=tk.CENTER,
                pady=theme.PAD_XL * 2,
            ).pack(fill=tk.X)
            return

        self._cartes: dict[str, Frame] = {}
        for i, (nom, nombre) in enumerate(sorted(par_nom.items())):
            self._cartes[nom] = self._creer_carte(nom, nombre, *divmod(i, COLONNES))
        for c in range(COLONNES):
            self._grille.columnconfigure(c, weight=1)

    def _creer_carte(self, nom: str, nombre: int, ligne: int, colonne: int) -> Frame:
        carte = Frame(self._grille, bg=theme.BG_CARD, highlightthickness=theme.BORDER_W)
        carte.configure(highlightbackground=theme.BORDER, highlightcolor=theme.BORDER)
        carte.grid(row=ligne, column=colonne, padx=theme.PAD_S, pady=theme.PAD_S, sticky=tk.NSEW)

        apercu = Label(carte, bg=theme.BG_BASE, width=12, height=6)
        image = self._repo.load_image(nom)
        if image is not None:
            try:
                carre = cv2.resize(image, (VIGNETTE, VIGNETTE))
                rgb = cv2.cvtColor(carre, cv2.COLOR_BGR2RGB)
                photo = ImageTk.PhotoImage(Image.fromarray(rgb))
                self._photos.append(photo)
                apercu.configure(image=photo, width=VIGNETTE, height=VIGNETTE)
            except Exception as exc:
                logger.warning("Vignette de '%s' non générée : %s", nom, exc)
        else:
            apercu.configure(text="SANS PHOTO", fg=theme.TEXT_SECONDARY, font=theme.FONT_BADGE())
        apercu.pack(padx=theme.PAD_S, pady=(theme.PAD_S, theme.PAD_XS))

        Label(
            carte,
            text=nom.upper(),
            bg=theme.BG_CARD,
            fg=theme.TEXT_PRIMARY,
            font=theme.FONT_HEADING(),
        ).pack()
        Label(
            carte,
            text=f"{nombre} ENCODAGE(S)",
            bg=theme.BG_CARD,
            fg=theme.TEXT_SECONDARY,
            font=theme.FONT_BADGE(),
        ).pack(pady=(0, theme.PAD_S))

        for widget in (carte, apercu, *carte.winfo_children()):
            widget.bind("<Button-1>", lambda _e, n=nom: self._selectionner(n))
        return carte

    def _selectionner(self, nom: str) -> None:
        self._selection = nom
        self._selection_var.set(f"SÉLECTION : {nom.upper()}")
        for autre, carte in self._cartes.items():
            actif = autre == nom
            couleur = theme.BRAND_ACCENT if actif else theme.BORDER
            carte.configure(highlightbackground=couleur, highlightcolor=couleur)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _enroler(self) -> None:
        from ..image_importer import ImageImporterApp

        fenetre = ImageImporterApp(self.winfo_toplevel())
        fenetre.focus()
        # Rafraîchir quand l'import se termine.
        fenetre.bind("<Destroy>", lambda _e: self.after(100, self.recharger))

    def _supprimer(self) -> None:
        if not self._selection:
            messagebox.showinfo(
                "Sélection", "Choisissez d'abord un visage.", parent=self.winfo_toplevel()
            )
            return
        nom = self._selection
        if not messagebox.askyesno(
            "Confirmation",
            f"Supprimer tous les encodages de « {nom} » ?",
            parent=self.winfo_toplevel(),
        ):
            return
        supprimes = self._repo.delete(nom)
        if not supprimes:
            messagebox.showerror(
                "Suppression",
                f"Aucun encodage trouvé pour « {nom} ».",
                parent=self.winfo_toplevel(),
            )
            return
        self.services.engine.force_refresh_encodings()
        self.recharger()
