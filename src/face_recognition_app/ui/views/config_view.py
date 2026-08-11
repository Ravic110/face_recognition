"""
config_view.py
Vue « Configuration » — caméras et accès à l'API.

Deux panneaux, comme dans les maquettes : la liste des caméras et leurs actions
à gauche, l'accès à l'API à droite. Le dialogue de configuration d'une caméra
reste modal — c'est une saisie ponctuelle avec validation, un des rares cas où
la modale est le bon outil.
"""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import messagebox, ttk

from ... import theme
from ..widgets import Frame, Label
from .base import AppServices, View, entete_panneau

logger = logging.getLogger(__name__)


class ConfigView(View):
    """Gestion des caméras et de l'accès distant."""

    TITRE = "CONFIGURATION"

    def __init__(self, parent, services: AppServices) -> None:
        super().__init__(parent, services)

    # ── Construction ──────────────────────────────────────────────────────────

    def construire(self) -> None:
        corps = tk.PanedWindow(
            self,
            orient=tk.HORIZONTAL,
            bg=theme.BORDER,
            sashwidth=theme.BORDER_W,
            sashrelief=tk.FLAT,
            bd=0,
        )
        corps.pack(fill=tk.BOTH, expand=True)

        gauche = Frame(corps, bg=theme.BG_BASE)
        corps.add(gauche, minsize=560)
        self._construire_cameras(gauche)

        droite = Frame(corps, bg=theme.BG_BASE)
        corps.add(droite, minsize=340)
        self._construire_api(droite)

    def _construire_cameras(self, parent) -> None:
        entete_panneau(parent, "CAMÉRAS CONFIGURÉES")

        barre = Frame(parent, bg=theme.BG_BASE, padx=theme.PAD_M, pady=theme.PAD_S)
        barre.pack(fill=tk.X)
        for libelle, commande, style in (
            ("＋ AJOUTER", self._ajouter, "success"),
            ("MODIFIER", self._modifier, "secondary-outline"),
            ("SUPPRIMER", self._supprimer, "danger-outline"),
            ("TESTER", self._tester, "secondary-outline"),
        ):
            ttk.Button(barre, text=libelle, command=commande, bootstyle=style).pack(
                side=tk.LEFT, padx=(0, theme.PAD_S)
            )

        cadre = Frame(parent, bg=theme.BG_BASE, padx=theme.PAD_M, pady=theme.PAD_S)
        cadre.pack(fill=tk.BOTH, expand=True)
        self._liste = ttk.Treeview(
            cadre,
            columns=("nom", "type", "source", "miroir", "etat"),
            show="headings",
            selectmode="browse",
        )
        for cle, titre, largeur, ancre in (
            ("nom", "NOM", 150, tk.W),
            ("type", "TYPE", 80, tk.CENTER),
            ("source", "SOURCE", 240, tk.W),
            ("miroir", "MIROIR", 70, tk.CENTER),
            ("etat", "ÉTAT", 90, tk.CENTER),
        ):
            self._liste.heading(cle, text=titre)
            self._liste.column(cle, width=largeur, anchor=ancre)
        defilement = ttk.Scrollbar(cadre, orient=tk.VERTICAL, command=self._liste.yview)
        self._liste.configure(yscrollcommand=defilement.set)
        defilement.pack(side=tk.RIGHT, fill=tk.Y)
        self._liste.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._liste.tag_configure("actif", foreground=theme.BRAND_ACCENT)
        self._liste.tag_configure("attente", foreground=theme.STATE_WARN)
        self._liste.tag_configure("inactif", foreground=theme.TEXT_SECONDARY)

        self.recharger()

    def _construire_api(self, parent) -> None:
        entete_panneau(parent, "ACCÈS À L'API")

        cadre = Frame(parent, bg=theme.BG_BASE, padx=theme.PAD_M, pady=theme.PAD_M)
        cadre.pack(fill=tk.BOTH, expand=True)

        ligne_etat = Frame(cadre, bg=theme.BG_BASE)
        ligne_etat.pack(fill=tk.X, pady=(0, theme.PAD_S))
        self._api_dot = Label(
            ligne_etat,
            text="●",
            bg=theme.BG_BASE,
            fg=theme.TEXT_SECONDARY,
            font=theme.FONT_SMALL(),
        )
        self._api_dot.pack(side=tk.LEFT, padx=(0, theme.PAD_XS))
        self._api_etat = tk.StringVar(value="")
        Label(
            ligne_etat,
            textvariable=self._api_etat,
            bg=theme.BG_BASE,
            fg=theme.TEXT_SECONDARY,
            font=theme.FONT_BADGE(),
        ).pack(side=tk.LEFT)

        Label(
            cadre,
            text="Envoyez cette clé dans l'en-tête X-API-Key.",
            bg=theme.BG_BASE,
            fg=theme.TEXT_SECONDARY,
            font=theme.FONT_SMALL(),
            anchor=tk.W,
            justify=tk.LEFT,
        ).pack(fill=tk.X, pady=(theme.PAD_S, theme.PAD_XS))

        self._cle_var = tk.StringVar(value=self.services.api.api_key)
        champ = ttk.Entry(cadre, textvariable=self._cle_var, font=theme.FONT_MONO())
        champ.configure(state="readonly")
        champ.pack(fill=tk.X)

        boutons = Frame(cadre, bg=theme.BG_BASE)
        boutons.pack(fill=tk.X, pady=theme.PAD_S)
        ttk.Button(
            boutons, text="COPIER LA CLÉ", command=self._copier, bootstyle="secondary-outline"
        ).pack(side=tk.LEFT)
        self._api_bouton = ttk.Button(
            boutons, text="DÉMARRER L'API", command=self._basculer_api, bootstyle="success-outline"
        )
        self._api_bouton.pack(side=tk.LEFT, padx=theme.PAD_S)

        self._url_var = tk.StringVar(value="")
        Label(
            cadre,
            textvariable=self._url_var,
            bg=theme.BG_BASE,
            fg=theme.TEXT_SECONDARY,
            font=theme.FONT_MONO(),
            anchor=tk.W,
        ).pack(fill=tk.X)

        avertissement = (
            "L'API écoute sur 127.0.0.1 : elle n'est joignable que depuis ce poste.\n"
            "Pour y accéder depuis un téléphone, définissez FR_API_HOST=0.0.0.0 —\n"
            "la clé devient alors le seul rempart.\n\n"
            "Le contrôle à distance (démarrer / arrêter la surveillance) est\n"
            "désactivé ; FR_API_ALLOW_CONTROL=true l'autorise."
        )
        Label(
            cadre,
            text=avertissement,
            bg=theme.BG_CARD,
            fg=theme.TEXT_SECONDARY,
            font=theme.FONT_SMALL(),
            justify=tk.LEFT,
            anchor=tk.W,
            padx=theme.PAD_S,
            pady=theme.PAD_S,
        ).pack(fill=tk.X, pady=(theme.PAD_M, 0))

        self._rafraichir_api()

    def on_show(self) -> None:
        if self._construite:
            self.recharger()
            self._rafraichir_api()

    # ── Caméras ───────────────────────────────────────────────────────────────

    def recharger(self) -> None:
        for ligne in self._liste.get_children():
            self._liste.delete(ligne)
        for cfg in self.services.cameras.list_configs():
            if self.services.cameras.is_running(cfg.uid):
                etat, tag = "ACTIF", "actif"
            elif not cfg.enabled:
                etat, tag = "DÉSACTIVÉ", "inactif"
            else:
                etat, tag = "ARRÊTÉ", "attente"
            self._liste.insert(
                "",
                tk.END,
                iid=cfg.uid,
                values=(
                    cfg.name,
                    "WEBCAM" if cfg.source_type == "webcam" else "IP",
                    str(cfg.source),
                    "OUI" if cfg.mirror else "NON",
                    etat,
                ),
                tags=(tag,),
            )

    def _uid_selectionne(self, action: str) -> str | None:
        selection = self._liste.selection()
        if not selection:
            messagebox.showinfo(
                "Sélection",
                f"Sélectionnez une caméra à {action}.",
                parent=self.winfo_toplevel(),
            )
            return None
        return selection[0]

    def _ajouter(self) -> None:
        from ..camera_config_dialog import CameraConfigDialog

        parent = self.winfo_toplevel()
        dlg = CameraConfigDialog(parent, title="Ajouter une caméra")
        parent.wait_window(dlg)
        if dlg.result:
            self.services.cameras.add_camera(dlg.result)
            if self.services.engine.is_running:
                self.services.cameras.start_camera(dlg.result.uid)
            self.recharger()

    def _modifier(self) -> None:
        from ..camera_config_dialog import CameraConfigDialog

        uid = self._uid_selectionne("modifier")
        if uid is None:
            return
        config = self.services.cameras.get_config(uid)
        if config is None:
            return
        parent = self.winfo_toplevel()
        dlg = CameraConfigDialog(parent, title="Modifier la caméra", config=config)
        parent.wait_window(dlg)
        if dlg.result:
            self.services.cameras.update_camera(dlg.result)
            self.recharger()

    def _supprimer(self) -> None:
        uid = self._uid_selectionne("supprimer")
        if uid is None:
            return
        config = self.services.cameras.get_config(uid)
        nom = config.name if config else uid
        if messagebox.askyesno(
            "Confirmation", f"Supprimer la caméra « {nom} » ?", parent=self.winfo_toplevel()
        ):
            self.services.cameras.remove_camera(uid)
            self.recharger()

    def _tester(self) -> None:
        uid = self._uid_selectionne("tester")
        if uid is None:
            return
        config = self.services.cameras.get_config(uid)
        if config is None:
            return

        import cv2

        capture = cv2.VideoCapture(config.source)
        ouverte = capture.isOpened()
        capture.release()
        if ouverte:
            messagebox.showinfo(
                "Test", f"✓ Connexion réussie à « {config.name} ».", parent=self.winfo_toplevel()
            )
        else:
            messagebox.showerror(
                "Test",
                f"✗ Impossible de se connecter à « {config.name} ».\nSource : {config.source}",
                parent=self.winfo_toplevel(),
            )

    # ── API ───────────────────────────────────────────────────────────────────

    def _copier(self) -> None:
        self.clipboard_clear()
        self.clipboard_append(self.services.api.api_key)

    def _basculer_api(self) -> None:
        api = self.services.api
        if api.is_running:
            api.stop()
        elif not api.start():
            messagebox.showerror(
                "API",
                f"Impossible d'écouter sur {api.url}.\nLe port est peut-être déjà utilisé.",
                parent=self.winfo_toplevel(),
            )
        self._rafraichir_api()

    def _rafraichir_api(self) -> None:
        api = self.services.api
        active = api.is_running
        self._api_dot.configure(fg=theme.BRAND_ACCENT if active else theme.TEXT_SECONDARY)
        self._api_etat.set("API ACTIVE" if active else "API HORS LIGNE")
        self._api_bouton.configure(
            text="ARRÊTER L'API" if active else "DÉMARRER L'API",
            bootstyle="danger-outline" if active else "success-outline",
        )
        self._url_var.set(f"{api.url}/api/status" if active else "")
