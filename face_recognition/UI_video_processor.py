import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox, Querybox
from PIL import Image, ImageTk
from tkinter import Canvas, Frame, Scrollbar, filedialog
from video_processor import process_video
from utils import save_face_encoding

class VideoImporterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Importateur Vidéo - Reconnaissance Faciale")
        self.root.geometry("700x600")
        self.style = ttk.Style("solar")  # Thème stylisé, vous pouvez essayer d'autres thèmes

        self.video_path = None
        self.tolerance = 0.5  # Tolérance pour la détection des visages

        # Widgets de l'interface utilisateur
        ttk.Label(root, text="Importateur Vidéo", font=("Helvetica", 20, "bold")).pack(pady=15)
        self.select_btn = ttk.Button(root, text="Sélectionner une vidéo", command=self.select_video, bootstyle=PRIMARY)
        self.select_btn.pack(pady=10)

        self.video_label = ttk.Label(root, text="Aucune vidéo sélectionnée", wraplength=600, bootstyle=INFO)
        self.video_label.pack(pady=10)

        self.process_btn = ttk.Button(root, text="Lancer le traitement", command=self.process_video, bootstyle=SUCCESS, state=DISABLED)
        self.process_btn.pack(pady=15)

        self.progress_bar = ttk.Progressbar(root, orient=HORIZONTAL, length=400, mode="determinate", bootstyle=INFO)
        self.progress_bar.pack(pady=20)

        self.log_label = ttk.Label(root, text="", wraplength=600, justify="left", bootstyle=SUCCESS)
        self.log_label.pack(pady=15)

    def select_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Vidéos", "*.mp4;*.avi;*.mov")])
        if self.video_path:
            self.video_label.config(text=f"Vidéo sélectionnée : {self.video_path}")
            self.process_btn.config(state=NORMAL)
        else:
            self.video_label.config(text="Aucune vidéo sélectionnée")

    def process_video(self):
        if not self.video_path:
            Messagebox.show_error("Erreur", "Veuillez sélectionner une vidéo avant de commencer.")
            return

        try:
            all_faces = []  # Liste pour collecter tous les visages détectés
            for faces in process_video(self.video_path, tolerance=self.tolerance):
                all_faces.extend(faces)

            if all_faces:
                self.show_faces_for_selection(all_faces)
            else:
                Messagebox.show_info("Aucun visage", "Aucun visage détecté dans la vidéo.")
        except Exception as e:
            Messagebox.show_error("Erreur", f"Une erreur s'est produite : {e}")

    def show_faces_for_selection(self, faces):
        """
        Affiche tous les visages détectés dans une seule fenêtre pour permettre à l'utilisateur
        de sélectionner ceux qu'il souhaite encoder.
        """
        # Fenêtre secondaire pour afficher les visages détectés
        selection_window = ttk.Toplevel(self.root)
        selection_window.title("Sélectionnez les visages à encoder")
        selection_window.geometry("850x650")

        # Canvas pour scroller si le nombre de visages dépasse la taille de la fenêtre
        canvas = Canvas(selection_window, bg="white", highlightthickness=0)
        frame = Frame(canvas, bg="white")
        scrollbar = ttk.Scrollbar(selection_window, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.create_window((0, 0), window=frame, anchor="nw")

        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        selected_faces = []  # Stocke les cases cochées et leurs encodages

        for i, (face_image, face_encoding) in enumerate(faces):
            # Convertir l'image OpenCV en image PIL
            img = Image.fromarray(face_image)
            img.thumbnail((100, 100))  # Taille uniforme des miniatures
            img_tk = ImageTk.PhotoImage(img)

            # Case à cocher pour sélectionner le visage
            var = ttk.IntVar()
            cb = ttk.Checkbutton(frame, image=img_tk, variable=var, bootstyle=INFO)
            cb.image = img_tk  # Nécessaire pour empêcher le garbage collection
            cb.grid(row=i // 5, column=i % 5, padx=10, pady=10)  # Affichage en grille
            selected_faces.append((var, face_encoding))

        def save_selected_faces():
            """
            Enregistrer les visages sélectionnés par l'utilisateur.
            """
            for var, face_encoding in selected_faces:
                if var.get() == 1:  # Si la case est cochée
                    name = Querybox.get_string("Nom requis", "Entrez le nom pour ce visage :")
                    if name:
                        save_face_encoding(name, face_encoding)
            selection_window.destroy()
            Messagebox.show_info("Encodage terminé", "Les visages sélectionnés ont été encodés.")

        ttk.Button(selection_window, text="Enregistrer", command=save_selected_faces, bootstyle=SUCCESS).pack(pady=15)

if __name__ == "__main__":
    root = ttk.Window(themename="solar")
    app = VideoImporterApp(root)
    root.mainloop()
