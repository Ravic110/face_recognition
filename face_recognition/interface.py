import tkinter as tk
from tkinter import ttk, messagebox

import use_webCam
import import_image
import import_video
import verify_if_exist

def launch_webcam():
    """Lancer la capture depuis la webcam."""
    try:
        name = input("Entrez le nom de la personne : ")
        encoder = use_webCam.SecureFaceEncoder()
        encoder.capture_face(name)
        messagebox.showinfo("Succès", "Capture depuis la webcam terminée !")
    except Exception as e:
        messagebox.showerror("Erreur", f"Une erreur s'est produite : {e}")

def import_image():
    """Importer une image et encoder le visage."""
    try:
        name = input("Entrez le nom de la personne : ")
        import_image.import_face_from_image(name)
        messagebox.showinfo("Succès", "Image importée et visage encodé avec succès !")
    except Exception as e:
        messagebox.showerror("Erreur", f"Une erreur s'est produite : {e}")

def import_video():
    """Importer une vidéo et encoder les visages."""
    try:
        import_video.import_video()
        messagebox.showinfo("Succès", "Vidéo importée et visages encodés avec succès !")
    except Exception as e:
        messagebox.showerror("Erreur", f"Une erreur s'est produite : {e}")

def verify_face():
    """Vérifier si un visage existe déjà."""
    try:
        verify_if_exist.import_and_check_face()
    except Exception as e:
        messagebox.showerror("Erreur", f"Une erreur s'est produite : {e}")

def show_help():
    """Afficher une boîte de dialogue avec des instructions d'utilisation."""
    help_text = (
        "1. Capture depuis la webcam : Prenez une photo pour encoder un visage.\n"
        "2. Importer une image : Chargez une image depuis votre ordinateur pour encoder un visage.\n"
        "3. Importer une vidéo : Analysez une vidéo pour détecter des visages.\n"
        "4. Vérifier un visage : Comparez un visage avec ceux déjà encodés."
    )
    messagebox.showinfo("Aide", help_text)

def main():
    """Créer l'interface principale."""
    root = tk.Tk()
    root.title("Gestion des Visages")

    # Titre principal
    ttk.Label(root, text="Gestion des Visages", font=("Arial", 20)).pack(pady=20)

    # Boutons d'options
    ttk.Button(root, text="Capture depuis la webcam", command=launch_webcam, width=40).pack(pady=10)
    ttk.Button(root, text="Importer une image", command=import_image, width=40).pack(pady=10)
    ttk.Button(root, text="Importer une vidéo", command=import_video, width=40).pack(pady=10)
    ttk.Button(root, text="Vérifier si un visage existe", command=verify_face, width=40).pack(pady=10)

    # Bouton d'aide
    ttk.Button(root, text="Aide", command=show_help, width=20).pack(pady=20)

    # Lancer l'interface
    root.mainloop()

if __name__ == "__main__":
    main()
