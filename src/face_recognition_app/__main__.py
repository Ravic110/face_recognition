import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

from face_recognition_app.ui import interface, import_image, video_importer


def launch_surveillance_dashboard(root):
    """
    Lance le tableau de bord en réutilisant le root ttkbootstrap existant.

    Le menu est caché (withdraw) et le dashboard s'ouvre comme Toplevel.
    Quand le dashboard se ferme, il détruit le root, ce qui termine l'app.
    """
    from face_recognition_app.ui.surveillance_dashboard import SurveillanceDashboard
    root.withdraw()
    SurveillanceDashboard(root)


def launch_image_interface(root):
    window = tk.Toplevel(root)
    interface.FaceRecognitionApp(window)


def launch_realtime_recognition(root):
    window = tk.Toplevel(root)
    app = interface.FaceRecognitionApp(window)
    window.after(0, app.start_camera)


def launch_video_importer(root):
    window = tk.Toplevel(root)
    video_importer.VideoImporterApp(window)


def launch_image_importer(root):
    """Lance le module d'import d'images avec interface graphique."""
    from face_recognition_app.ui.image_importer import ImageImporterApp
    ImageImporterApp(root)


def launch_import_image_cli():
    import_image.import_face_from_image()


def main():
    root = ttk.Window(themename="solar")
    root.title("Système de Reconnaissance Faciale")
    root.geometry("520x500")
    root.resizable(False, False)

    # En-tête
    header_frame = ttk.Frame(root, padding=(20, 20, 20, 10))
    header_frame.pack(fill="x")

    ttk.Label(
        header_frame,
        text="Surveillance Intelligente",
        font=("Helvetica", 20, "bold"),
        bootstyle=PRIMARY,
    ).pack()

    ttk.Label(
        header_frame,
        text="Choisissez un mode pour commencer",
        font=("Helvetica", 10),
        bootstyle=SECONDARY,
    ).pack(pady=(4, 0))

    ttk.Separator(root, orient="horizontal").pack(fill="x", padx=20, pady=8)

    # Boutons
    btn_frame = ttk.Frame(root, padding=(30, 5, 30, 20))
    btn_frame.pack(expand=True, fill="both")

    buttons = [
        # ── Nouveau mode principal ──────────────────────────────────────────
        ("Tableau de bord Surveillance (multi-caméras)",
         lambda: launch_surveillance_dashboard(root), SUCCESS),
        # ── Modules existants ───────────────────────────────────────────────
        ("Reconnaissance faciale (image)", lambda: launch_image_interface(root), PRIMARY),
        ("Reconnaissance en temps réel (Webcam)", lambda: launch_realtime_recognition(root), PRIMARY),
        ("Importer une vidéo", lambda: launch_video_importer(root), INFO),
        ("Importer des images (enregistrer des visages)", lambda: launch_image_importer(root), INFO),
        ("Quitter", root.destroy, DANGER),
    ]

    for text, command, style in buttons:
        ttk.Button(
            btn_frame,
            text=text,
            command=command,
            width=42,
            bootstyle=style,
        ).pack(pady=5, fill="x")

    root.mainloop()


if __name__ == "__main__":
    main()
