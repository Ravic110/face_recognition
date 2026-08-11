"""
widgets.py
Enveloppes des widgets Tk classiques, soustraits au style automatique.

`ttkbootstrap` enveloppe le constructeur de chaque widget `tk` et lui réapplique
sa propre palette juste après l'initialisation — `bg` et `fg` passés à la
construction sont donc écrasés en silence, ce qui aplatit tout l'étagement des
fonds. Le mécanisme se trouve dans `Bootstyle.override_tk_widget_constructor`.

Il expose heureusement une sortie : le mot-clé `autostyle`. À `False`, le widget
n'est ni abonné aux changements de thème ni restylé, et les couleurs demandées
tiennent.

On l'applique ici une fois pour toutes, plutôt que de répéter `autostyle=False`
sur chaque appel. Les widgets `ttk.*` ne sont pas concernés : eux doivent rester
pilotés par le thème.
"""

from __future__ import annotations

import tkinter as tk
from typing import Any

__all__ = ["Canvas", "Frame", "Label", "Text", "sans_autostyle"]


def sans_autostyle(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Ajoute `autostyle=False`, pour les sous-classes qui appellent `super()`."""
    return {"autostyle": False, **kwargs}


class Frame(tk.Frame):
    """`tk.Frame` dont les couleurs ne sont pas écrasées par le thème."""

    def __init__(self, master=None, **kwargs: Any) -> None:
        super().__init__(master, **sans_autostyle(kwargs))


class Label(tk.Label):
    """`tk.Label` dont les couleurs ne sont pas écrasées par le thème."""

    def __init__(self, master=None, **kwargs: Any) -> None:
        super().__init__(master, **sans_autostyle(kwargs))


class Canvas(tk.Canvas):
    """`tk.Canvas` dont les couleurs ne sont pas écrasées par le thème."""

    def __init__(self, master=None, **kwargs: Any) -> None:
        super().__init__(master, **sans_autostyle(kwargs))


class Text(tk.Text):
    """`tk.Text` dont les couleurs ne sont pas écrasées par le thème."""

    def __init__(self, master=None, **kwargs: Any) -> None:
        super().__init__(master, **sans_autostyle(kwargs))
