"""
alerts_view.py
Vue « Alertes » — configuration des canaux de notification.

Anciennement une boîte modale ; elle vit désormais dans la zone centrale.
L'enregistrement ne ferme donc plus rien : il confirme sur place.

Les filtres « alerter sur visage connu / inconnu » restent affichés ici : ils
existent en double avec le profil de surveillance, et c'est toujours la version
d'`AlertConfig` qui décide. La fusion prévue dans la spec appartient à la
refonte d'`AlertManager`, encore à venir.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ... import theme
from ..widgets import Frame, Label
from .base import View, entete_panneau


class AlertsView(View):
    """Configuration des canaux d'alerte."""

    TITRE = "ALERTES"

    def construire(self) -> None:
        entete_panneau(self, "CONFIGURATION DES ALERTES")

        self._mgr = self.services.alerts
        cfg = self._mgr.config
        alert_mgr = self._mgr

        cadre = Frame(self, bg=theme.BG_BASE, padx=theme.PAD_L, pady=theme.PAD_M)
        cadre.pack(fill=tk.BOTH, expand=True)

        pad = {"padx": theme.PAD_M, "pady": theme.PAD_S}
        nb = ttk.Notebook(cadre)
        nb.pack(fill=tk.BOTH, expand=True)

        # ── Onglet Bureau ──────────────────────────────────────────────────────
        tab_desk = Frame(nb, bg=theme.BG_CARD)
        nb.add(tab_desk, text="Bureau")
        self._desk_en = tk.BooleanVar(value=cfg.desktop_enabled)
        ttk.Checkbutton(
            tab_desk, text="Activer les notifications bureau", variable=self._desk_en
        ).pack(anchor=tk.W, **pad)
        ttk.Button(tab_desk, text="Tester", command=lambda: alert_mgr.test_desktop()).pack(
            anchor=tk.W, padx=10
        )

        # ── Onglet Email ───────────────────────────────────────────────────────
        tab_mail = Frame(nb, bg=theme.BG_CARD)
        nb.add(tab_mail, text="Email")
        self._mail_en = tk.BooleanVar(value=cfg.email_enabled)
        ttk.Checkbutton(tab_mail, text="Activer les alertes email", variable=self._mail_en).pack(
            anchor=tk.W, **pad
        )
        fields = [
            ("Serveur SMTP", "smtp_host"),
            ("Port", "smtp_port"),
            ("Utilisateur", "smtp_user"),
            ("Mot de passe", "smtp_password"),
            ("Destinataires (virgule)", "email_recipients"),
        ]
        self._mail_vars: dict = {}
        for label, key in fields:
            row = Frame(tab_mail, bg=theme.BG_CARD)
            row.pack(fill=tk.X, padx=10, pady=2)
            Label(
                row,
                text=label,
                width=22,
                anchor=tk.W,
                bg=theme.BG_CARD,
                fg=theme.TEXT_SECONDARY,
                font=theme.FONT_SMALL(),
            ).pack(side=tk.LEFT)
            val = getattr(cfg, key)
            if isinstance(val, list):
                val = ", ".join(val)
            var = tk.StringVar(value=str(val))
            self._mail_vars[key] = var
            ttk.Entry(row, textvariable=var, width=28, show="*" if "password" in key else "").pack(
                side=tk.LEFT
            )

        # ── Onglet Webhook ────────────────────────────────────────────────────
        tab_wh = Frame(nb, bg=theme.BG_CARD)
        nb.add(tab_wh, text="Webhook")
        self._wh_en = tk.BooleanVar(value=cfg.webhook_enabled)
        ttk.Checkbutton(tab_wh, text="Activer le webhook", variable=self._wh_en).pack(
            anchor=tk.W, **pad
        )
        for label, key in [("URL", "webhook_url"), ("Secret", "webhook_secret")]:
            row = Frame(tab_wh, bg=theme.BG_CARD)
            row.pack(fill=tk.X, padx=10, pady=2)
            Label(
                row,
                text=label,
                width=10,
                anchor=tk.W,
                bg=theme.BG_CARD,
                fg=theme.TEXT_SECONDARY,
                font=theme.FONT_SMALL(),
            ).pack(side=tk.LEFT)
            var = tk.StringVar(value=getattr(cfg, key))
            setattr(self, f"_wh_{key}", var)
            ttk.Entry(row, textvariable=var, width=36).pack(side=tk.LEFT)

        # ── Onglet Filtres ────────────────────────────────────────────────────
        tab_f = Frame(nb, bg=theme.BG_CARD)
        nb.add(tab_f, text="Filtres")
        self._unk_var = tk.BooleanVar(value=cfg.alert_on_unknown)
        self._kn_var = tk.BooleanVar(value=cfg.alert_on_known)
        self._cooldown_var = tk.StringVar(value=str(cfg.cooldown_seconds))
        ttk.Checkbutton(tab_f, text="Alerter sur visage inconnu", variable=self._unk_var).pack(
            anchor=tk.W, **pad
        )
        ttk.Checkbutton(tab_f, text="Alerter sur visage connu", variable=self._kn_var).pack(
            anchor=tk.W, **pad
        )
        row = Frame(tab_f, bg=theme.BG_CARD)
        row.pack(anchor=tk.W, **pad)
        Label(
            row,
            text="ANTI-SPAM (SECONDES)",
            bg=theme.BG_CARD,
            fg=theme.TEXT_SECONDARY,
            font=theme.FONT_BADGE(),
        ).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self._cooldown_var, width=6).pack(side=tk.LEFT, padx=4)

        # Boutons
        btn_frame = Frame(self)
        btn_frame.pack(fill=tk.X, padx=8, pady=(0, 10))
        ttk.Button(btn_frame, text="Annuler", command=self.destroy).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btn_frame, text="Sauvegarder", command=self._save).pack(side=tk.RIGHT)

    def _save(self) -> None:
        cfg = self._mgr.config
        cfg.desktop_enabled = self._desk_en.get()
        cfg.email_enabled = self._mail_en.get()
        cfg.smtp_host = self._mail_vars["smtp_host"].get()
        cfg.smtp_port = int(self._mail_vars["smtp_port"].get() or "587")
        cfg.smtp_user = self._mail_vars["smtp_user"].get()
        cfg.smtp_password = self._mail_vars["smtp_password"].get()
        cfg.email_recipients = [
            r.strip() for r in self._mail_vars["email_recipients"].get().split(",") if r.strip()
        ]
        cfg.webhook_enabled = self._wh_en.get()
        cfg.webhook_url = self._wh_webhook_url.get()
        cfg.webhook_secret = self._wh_webhook_secret.get()
        cfg.alert_on_unknown = self._unk_var.get()
        cfg.alert_on_known = self._kn_var.get()
        try:
            cfg.cooldown_seconds = float(self._cooldown_var.get())
        except ValueError:
            # Champ non numérique : on garde la valeur précédente et on le dit.
            self._cooldown_var.set(str(cfg.cooldown_seconds))
        self._mgr.config = cfg
        self._retour.set("Configuration enregistrée.")
        self.after(4000, lambda: self._retour.set(""))
