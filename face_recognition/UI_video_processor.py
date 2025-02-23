import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox, Querybox
from PIL import Image, ImageTk
from tkinter import Canvas, Frame, Scrollbar, filedialog
from video_processor import process_video
from utils import save_face_encoding, load_existing_encodings, is_duplicate
import threading
import cv2
import face_recognition

class VideoImporterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Importateur Vidéo - Reconnaissance Faciale")
        self.root.geometry("900x700")
        self.style = ttk.Style("solar")

        self.video_path = None
        self.tolerance = 0.5  # Tolérance pour la détection des visages
        self.frame_skip = 30  # Nombre de frames à sauter
        self.existing_encodings = load_existing_encodings()  # Charger les encodages existants

        # Widgets de l'interface utilisateur
        ttk.Label(root, text="Importateur Vidéo", font=("Helvetica", 20, "bold")).pack(pady=15)
        self.select_btn = ttk.Button(root, text="Sélectionner une vidéo", command=self.select_video, bootstyle=PRIMARY)
        self.select_btn.pack(pady=10)

        self.video_label = ttk.Label(root, text="Aucune vidéo sélectionnée", wraplength=600, bootstyle=INFO)
        self.video_label.pack(pady=10)

        # Configuration de frame_skip
        ttk.Label(root, text="Saut de frames (frame_skip):", bootstyle=INFO).pack(pady=5)
        self.frame_skip_entry = ttk.Entry(root, width=10)
        self.frame_skip_entry.insert(0, "30")
        self.frame_skip_entry.pack(pady=5)

        self.process_btn = ttk.Button(root, text="Lancer le traitement", command=self.start_processing, bootstyle=SUCCESS, state=DISABLED)
        self.process_btn.pack(pady=15)

        # Lecteur vidéo pour la prévisualisation
        self.video_player = ttk.Label(root, text="Prévisualisation vidéo", bootstyle=INFO)
        self.video_player.pack(pady=10)

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

    def start_processing(self):
        """Démarre le traitement vidéo dans un thread séparé."""
        try:
            self.frame_skip = int(self.frame_skip_entry.get())
            if self.frame_skip <= 0:
                raise ValueError("Le saut de frames doit être un entier positif.")
        except ValueError as e:
            Messagebox.show_error("Erreur", f"Valeur de frame_skip invalide : {e}")
            return

        # Désactiver les boutons pendant le traitement
        self.select_btn.config(state=DISABLED)
        self.process_btn.config(state=DISABLED)
        self.progress_bar["value"] = 0

        # Démarrer le traitement dans un thread séparé
        threading.Thread(target=self.process_video, daemon=True).start()

    def process_video(self):
        """Traite la vidéo et met à jour l'interface utilisateur."""
        if not self.video_path:
            Messagebox.show_error("Erreur", "Veuillez sélectionner une vidéo avant de commencer.")
            return

        try:
            cap = cv2.VideoCapture(self.video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.progress_bar["maximum"] = frame_count

            all_faces = []  # Liste pour collecter tous les visages détectés
            processed_frames = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frames += 1
                if processed_frames % self.frame_skip != 0:
                    continue

                # Afficher la frame dans le lecteur vidéo
                self.update_video_player(frame)

                # Traiter la frame pour détecter les visages
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                faces = []
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    face_image = frame[top:bottom, left:right]
                    faces.append((face_image, face_encoding))

                all_faces.extend(faces)
                self.progress_bar["value"] += self.frame_skip
                self.root.update_idletasks()  # Rafraîchir l'interface utilisateur

            if all_faces:
                # Regrouper les visages similaires
                unique_faces = self.group_similar_faces(all_faces)
                self.show_faces_for_selection(unique_faces)
            else:
                Messagebox.show_info("Aucun visage", "Aucun visage détecté dans la vidéo.")
        except Exception as e:
            Messagebox.show_error("Erreur", f"Une erreur s'est produite : {e}")
        finally:
            cap.release()
            # Réactiver les boutons après le traitement
            self.select_btn.config(state=NORMAL)
            self.process_btn.config(state=NORMAL)

    def update_video_player(self, frame):
        """Met à jour le lecteur vidéo avec une nouvelle frame."""
        # Convertir l'image OpenCV (BGR) en image PIL (RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img.thumbnail((640, 360))  # Redimensionner l'image pour l'affichage
        img_tk = ImageTk.PhotoImage(img)

        # Mettre à jour le lecteur vidéo
        self.video_player.config(image=img_tk)
        self.video_player.image = img_tk  # Empêcher le garbage collection

    def group_similar_faces(self, faces):
        """Regroupe les visages similaires en utilisant leur encodage facial."""
        unique_faces = []
        for face_image, face_encoding in faces:
            # Vérifier si le visage est déjà dans unique_faces
            is_new = True
            for existing_image, existing_encoding, _ in unique_faces:
                if self.are_faces_similar(face_encoding, existing_encoding):
                    is_new = False
                    break

            # Vérifier si le visage est déjà enregistré dans la base de données
            existing_name = is_duplicate(face_encoding, self.existing_encodings, self.tolerance)
            if existing_name:
                unique_faces.append((face_image, face_encoding, existing_name))
            elif is_new:
                unique_faces.append((face_image, face_encoding, None))

        return unique_faces

    def are_faces_similar(self, encoding1, encoding2):
        """Compare deux encodages faciaux pour déterminer s'ils sont similaires."""
        from face_recognition import compare_faces
        return compare_faces([encoding1], encoding2, tolerance=self.tolerance)[0]

    def show_faces_for_selection(self, faces):
        """Affiche les visages détectés pour sélection."""
        selection_window = ttk.Toplevel(self.root)
        selection_window.title("Sélectionnez les visages à encoder")
        selection_window.geometry("850x650")

        # Canvas pour scroller
        canvas = Canvas(selection_window, bg="white", highlightthickness=0)
        frame = Frame(canvas, bg="white")
        scrollbar = ttk.Scrollbar(selection_window, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.create_window((0, 0), window=frame, anchor="nw")

        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        for i, (face_image, face_encoding, existing_name) in enumerate(faces):
            # Convertir l'image OpenCV (BGR) en image PIL (RGB)
            face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(face_image_rgb)
            img.thumbnail((100, 100))  # Taille uniforme des miniatures
            img_tk = ImageTk.PhotoImage(img)

            # Créer un conteneur pour l'image et le nom
            container = ttk.Frame(frame)
            container.grid(row=i // 5, column=i % 5, padx=10, pady=10)

            # Afficher le nom du visage s'il est déjà enregistré
            if existing_name:
                name_label = ttk.Label(container, text=existing_name, bootstyle=INFO)
                name_label.pack()

            # Afficher l'image du visage
            label = ttk.Label(container, image=img_tk)
            label.image = img_tk  # Empêcher le garbage collection
            label.pack()

            # Associer un événement de clic à l'image
            if not existing_name:  # Seulement pour les nouveaux visages
                label.bind("<Button-1>", lambda e, fe=face_encoding: self.prompt_for_name(fe, selection_window))

        ttk.Button(selection_window, text="Fermer", command=selection_window.destroy, bootstyle=DANGER).pack(pady=15)

    def prompt_for_name(self, face_encoding, selection_window):
        """Affiche un pop-up pour demander le nom du visage."""
        name = Querybox.get_string("Nom requis", "Entrez le nom pour ce visage :")
        if name:
            save_face_encoding(name, face_encoding)
            Messagebox.show_info("Succès", f"Le visage '{name}' a été enregistré.")
            selection_window.destroy()  # Fermer la fenêtre de sélection après l'enregistrement

if __name__ == "__main__":
    root = ttk.Window(themename="solar")
    app = VideoImporterApp(root)
    root.mainloop()