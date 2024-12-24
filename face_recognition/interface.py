import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import os
import face_recognition
import json
from datetime import datetime
from tkinter.filedialog import askopenfilename
from threading import Thread
from config import ENCODED_DIR, META_FILE
import import_image


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconnaissance Faciale")

        # Définir une taille minimale
        self.root.minsize(1024, 768)

        # Obtenir les dimensions de l'écran
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # Calculer la taille optimale (80% de l'écran)
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)

        # Centrer la fenêtre
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Permettre le redimensionnement
        self.root.resizable(True, True)

        # Configuration du grid pour qu'il s'adapte
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.current_image = None
        self.current_image_tk = None
        self.face_locations = None
        self.face_encodings = None
        self.selected_face_index = None

        self.setup_ui()

    def verify_face(self, face_encoding):
        """
        Vérifier si le visage existe avec une meilleure précision
        """
        encoded_faces = self.load_all_encodings()
        best_match = None
        min_distance = float('inf')

        for name, known_encoding in encoded_faces.items():
            # Calculer la distance entre les encodages
            face_distance = face_recognition.face_distance([known_encoding], face_encoding)[0]

            # Si la distance est inférieure au seuil et c'est la plus petite trouvée
            if face_distance < 0.5 and face_distance < min_distance:
                min_distance = face_distance
                best_match = name

        return (best_match is not None), best_match

    def load_all_encodings(self):
        encoded_faces = {}
        for file_name in os.listdir(ENCODED_DIR):
            if file_name.endswith(".json") and file_name != os.path.basename(META_FILE):
                with open(os.path.join(ENCODED_DIR, file_name), 'r') as file:
                    data = json.load(file)
                    if 'name' in data and 'encoding' in data:
                        encoded_faces[data['name']] = np.array(data['encoding'])
        return encoded_faces

    def setup_ui(self):
        # Frame principale avec poids pour l'expansion
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.main_frame.grid_rowconfigure(1, weight=1)  # Ligne du canvas
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Zone des boutons en haut
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.grid(row=0, column=0, pady=(0, 10), sticky="ew")

        self.load_button = ttk.Button(
            self.button_frame,
            text="Charger une image",
            command=self.load_image,
            width=20
        )
        self.load_button.pack(side="left", padx=5)

        # Canvas redimensionnable
        self.canvas = tk.Canvas(
            self.main_frame,
            bg='gray85',
            width=800,
            height=600
        )
        self.canvas.grid(row=1, column=0, sticky="nsew", pady=10)

        # Frame d'information
        self.info_frame = ttk.LabelFrame(
            self.main_frame,
            text="Informations",
            padding="10"
        )
        self.info_frame.grid(row=2, column=0, sticky="ew", pady=(10, 5))

        self.info_label = ttk.Label(
            self.info_frame,
            text="Aucune image chargée"
        )
        self.info_label.grid(row=0, column=0, sticky="w")

        # Frame de sélection et d'enregistrement côte à côte
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.grid(row=3, column=0, sticky="ew", pady=5)
        self.control_frame.grid_columnconfigure(0, weight=1)
        self.control_frame.grid_columnconfigure(1, weight=1)

        # Frame de sélection (à gauche)
        self.selection_frame = ttk.LabelFrame(
            self.control_frame,
            text="Sélection du visage",
            padding="10"
        )
        self.selection_frame.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        self.face_var = tk.StringVar(value="")
        self.face_entry = ttk.Entry(
            self.selection_frame,
            textvariable=self.face_var,
            state="disabled"
        )
        self.face_entry.pack(side="left", padx=5)

        self.select_button = ttk.Button(
            self.selection_frame,
            text="Sélectionner ce visage",
            command=self.select_face,
            state="disabled"
        )
        self.select_button.pack(side="left", padx=5)

        # Frame d'enregistrement (à droite)
        self.name_frame = ttk.LabelFrame(
            self.control_frame,
            text="Enregistrement",
            padding="10"
        )
        self.name_frame.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        ttk.Label(self.name_frame, text="Nom:").pack(side="left", padx=5)
        self.name_entry = ttk.Entry(self.name_frame)
        self.name_entry.pack(side="left", padx=5)

        self.save_button = ttk.Button(
            self.name_frame,
            text="Enregistrer",
            command=self.save_face,
            state="disabled"
        )
        self.save_button.pack(side="left", padx=5)

        # Barre de progression
        self.progress = ttk.Progressbar(
            self.main_frame,
            mode='indeterminate'
        )
        self.progress.grid(row=4, column=0, sticky="ew", pady=10)

    def load_image(self):
        try:
            image_path = askopenfilename(
                title="Sélectionnez une image",
                filetypes=[("Images", "*.jpg;*.jpeg;*.png")]
            )

            if not image_path:
                return

            self.current_image = cv2.imread(image_path)
            if self.current_image is None:
                raise ValueError("Impossible de charger l'image")

            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            self.progress.start()
            Thread(target=self.process_image, daemon=True).start()

        except Exception as e:
            messagebox.showerror("Erreur", str(e))

    def process_image(self):
        try:
            image_resized = import_image.resize_image_to_fit_screen(
                self.current_image,
                max_width=self.canvas.winfo_width(),
                max_height=self.canvas.winfo_height()
            )

            self.face_locations = face_recognition.face_locations(image_resized)
            self.face_encodings = face_recognition.face_encodings(image_resized, self.face_locations)

            for i, encoding in enumerate(self.face_encodings):
                exists, known_name = self.verify_face(encoding)
                top, right, bottom, left = self.face_locations[i]

                if exists:
                    # Visage connu - Rouge avec nom
                    cv2.rectangle(image_resized, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(image_resized, f"{known_name}", (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                else:
                    # Nouveau visage - Vert
                    cv2.rectangle(image_resized, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(image_resized, f"Nouveau {i + 1}", (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            self.root.after(0, self.update_display, image_resized)

        except Exception as e:
            self.root.after(0, messagebox.showerror, "Erreur", str(e))
        finally:
            self.root.after(0, self.progress.stop)

    def update_display(self, image):
        # Obtenir les dimensions du canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # Créer l'image Tkinter
        self.current_image_tk = ImageTk.PhotoImage(Image.fromarray(image))

        # Calculer les coordonnées pour centrer l'image
        x = canvas_width // 2
        y = canvas_height // 2

        # Effacer le canvas et afficher l'image centrée
        self.canvas.delete("all")
        self.canvas.create_image(x, y, anchor=tk.CENTER, image=self.current_image_tk)

        if self.face_locations:
            self.info_label.config(text=f"{len(self.face_locations)} visage(s) détecté(s)")
            self.face_entry.config(state="normal")
            self.select_button.config(state="normal")
            self.face_var.set("")
        else:
            self.info_label.config(text="Aucun visage détecté")

    def select_face(self):
        try:
            face_num = int(self.face_var.get())
            if 1 <= face_num <= len(self.face_locations):
                self.selected_face_index = face_num - 1
                exists, _ = self.verify_face(self.face_encodings[self.selected_face_index])

                if exists:
                    messagebox.showwarning("Attention", "Ce visage est déjà enregistré")
                    return

                self.save_button.config(state="normal")
                self.name_entry.focus()
            else:
                messagebox.showwarning("Attention", "Numéro de visage invalide")
        except ValueError:
            messagebox.showwarning("Attention", "Veuillez entrer un numéro de visage valide")

    def save_face(self):
        if self.selected_face_index is None:
            return

        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Attention", "Veuillez entrer un nom")
            return

        try:
            import_image.save_face_encoding(name, self.face_encodings[self.selected_face_index])
            messagebox.showinfo("Succès", f"Visage de {name} enregistré avec succès")
            self.name_entry.delete(0, tk.END)
            self.face_var.set("")
            self.save_button.config(state="disabled")
        except Exception as e:
            messagebox.showerror("Erreur", str(e))


def main():
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()