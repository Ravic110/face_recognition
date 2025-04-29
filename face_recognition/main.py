import tkinter as tk
from tkinter import ttk
import interface
import video_processor

def launch_image_interface():
    """
    Lance l'interface graphique de traitement d'images statiques.
    """
    root = tk.Tk()
    app = interface.FaceRecognitionApp(root)
    root.mainloop()

def launch_realtime_recognition():
    """
    Lance la reconnaissance faciale en temps réel avec webcam.
    """
    video_processor.run_realtime_recognition()

def main():
    # Fenêtre principale
    root = tk.Tk()
    root.title("Système de Reconnaissance Faciale")

    root.geometry("400x300")
    root.resizable(False, False)

    frame = ttk.Frame(root, padding=20)
    frame.pack(expand=True)

    ttk.Label(frame, text="Que souhaitez-vous faire ?", font=("Arial", 14)).pack(pady=10)

    # Bouton pour l'interface image
    ttk.Button(
        frame,
        text="Charger une image",
        command=lambda: [root.destroy(), launch_image_interface()],
        width=30
    ).pack(pady=10)

    # Bouton pour la caméra temps réel
    ttk.Button(
        frame,
        text="Reconnaissance en temps réel (Webcam)",
        command=lambda: [root.destroy(), launch_realtime_recognition()],
        width=30
    ).pack(pady=10)

    # Quitter
    ttk.Button(
        frame,
        text="Quitter",
        command=root.destroy,
        width=30
    ).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
