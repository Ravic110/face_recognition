import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import threading
from typing import Optional, List, Dict, Tuple
import logging
import os
from datetime import datetime
import numpy as np
from dataclasses import dataclass
import face_recognition
from collections import defaultdict

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='video_import.log'
)


@dataclass
class DetectedFace:
    frame_number: int
    location: Tuple[int, int, int, int]
    encoding: np.ndarray
    quality: float
    image: np.ndarray


class FaceCollector:
    def __init__(self):
        self.detected_faces: List[DetectedFace] = []
        self.selected_faces: Dict[str, List[DetectedFace]] = defaultdict(list)

    def collect_faces(self, frame: np.ndarray, frame_number: int) -> List[Tuple[int, int, int, int]]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for location, encoding in zip(face_locations, face_encodings):
            top, right, bottom, left = location
            face_img = frame[top:bottom, left:right]
            quality = self._assess_quality(face_img)

            if quality > 0.15:
                self.detected_faces.append(DetectedFace(
                    frame_number=frame_number,
                    location=location,
                    encoding=encoding,
                    quality=quality,
                    image=face_img.copy()
                ))

        return face_locations

    def _assess_quality(self, face_img: np.ndarray) -> float:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.sum()
        contrast_score = np.std(hist_norm) * 100
        return (blur_score * 0.7 + contrast_score * 0.3) / 100

    def assign_face_to_person(self, face_idx: int, person_name: str):
        if 0 <= face_idx < len(self.detected_faces):
            face = self.detected_faces[face_idx]
            self.selected_faces[person_name].append(face)

    def get_best_faces_for_person(self, person_name: str, max_faces: int = 5) -> List[np.ndarray]:
        if person_name not in self.selected_faces:
            return []

        faces = self.selected_faces[person_name]
        faces.sort(key=lambda x: x.quality, reverse=True)
        return [face.encoding for face in faces[:max_faces]]

    def clear_collections(self):
        self.detected_faces.clear()
        self.selected_faces.clear()


class VideoImportInterface:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Import de Vidéo - Détection et Sélection de Visages")
        self.root.geometry("800x600")

        # Variables
        self.video_path: Optional[str] = None
        self.processing = False
        self.preview_thread = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.show_preview = tk.BooleanVar(value=True)
        self.preview_interval = 50
        self.last_preview_time = 0
        self.start_time = None
        self.collector = None

        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Zone de prévisualisation
        self.preview_frame = ttk.LabelFrame(main_frame, text="Prévisualisation", padding="5")
        self.preview_frame.grid(row=0, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.preview_label = ttk.Label(self.preview_frame)
        self.preview_label.grid(row=0, column=0, pady=5)

        # Contrôles
        controls_frame = ttk.LabelFrame(main_frame, text="Contrôles", padding="5")
        controls_frame.grid(row=1, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

        self.select_button = ttk.Button(
            controls_frame,
            text="Sélectionner une vidéo",
            command=self.select_video
        )
        self.select_button.grid(row=0, column=0, padx=5, pady=5)

        self.pause_button = ttk.Button(
            controls_frame,
            text="Pause",
            command=self.toggle_preview,
            state=tk.DISABLED
        )
        self.pause_button.grid(row=0, column=1, padx=5, pady=5)

        ttk.Checkbutton(
            controls_frame,
            text="Afficher la prévisualisation pendant le traitement",
            variable=self.show_preview
        ).grid(row=1, column=0, columnspan=2, pady=5)

        self.process_button = ttk.Button(
            controls_frame,
            text="Détecter les visages",
            command=self.start_processing,
            state=tk.DISABLED
        )
        self.process_button.grid(row=2, column=0, columnspan=2, pady=5)

        # Barre de progression
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            main_frame,
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        # Label d'état
        self.status_label = ttk.Label(main_frame, text="En attente d'une vidéo...")
        self.status_label.grid(row=3, column=0, columnspan=2, pady=5)

        # Frame pour les statistiques
        self.stats_frame = ttk.LabelFrame(main_frame, text="Statistiques", padding="5")
        self.stats_frame.grid(row=4, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

        self.faces_found_label = ttk.Label(self.stats_frame, text="Visages détectés : 0")
        self.faces_found_label.grid(row=0, column=0, padx=5)

        self.processing_time_label = ttk.Label(self.stats_frame, text="Temps écoulé : 00:00")
        self.processing_time_label.grid(row=0, column=1, padx=5)

        # Configuration du redimensionnement
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

    def select_video(self):
        filetypes = [("Vidéos", "*.mp4;*.avi;*.mkv;*.mov")]
        self.video_path = filedialog.askopenfilename(
            title="Sélectionnez une vidéo",
            filetypes=filetypes
        )

        if self.video_path and os.path.exists(self.video_path):
            self.status_label.config(text=f"Vidéo sélectionnée : {os.path.basename(self.video_path)}")
            self.process_button.config(state=tk.NORMAL)
            self.pause_button.config(state=tk.NORMAL)
            self.start_preview()
        else:
            self.video_path = None
            self.status_label.config(text="Veuillez sélectionner une vidéo valide")
            self.process_button.config(state=tk.DISABLED)
            self.pause_button.config(state=tk.DISABLED)

    def start_preview(self):
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Erreur", "Impossible d'ouvrir la vidéo")
            return

        def update_preview():
            while self.cap is not None and self.cap.isOpened():
                current_time = datetime.now().timestamp() * 1000
                if current_time - self.last_preview_time < self.preview_interval:
                    continue

                ret, frame = self.cap.read()
                if ret:
                    height, width = frame.shape[:2]
                    max_size = 400
                    if height > max_size or width > max_size:
                        scale = max_size / max(height, width)
                        frame = cv2.resize(frame, None, fx=scale, fy=scale)

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    img_tk = ImageTk.PhotoImage(image=img)
                    self.preview_label.config(image=img_tk)
                    self.preview_label.image = img_tk

                    self.last_preview_time = current_time
                else:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                if not self.show_preview.get():
                    break

                self.root.update_idletasks()

        self.preview_thread = threading.Thread(target=update_preview, daemon=True)
        self.preview_thread.start()

    def toggle_preview(self):
        if self.show_preview.get():
            self.show_preview.set(False)
            self.pause_button.config(text="Reprendre")
        else:
            self.show_preview.set(True)
            self.pause_button.config(text="Pause")
            self.start_preview()

    def start_processing(self):
        if not self.video_path:
            messagebox.showerror("Erreur", "Veuillez sélectionner une vidéo")
            return

        self.processing = True
        self.collector = FaceCollector()
        self.start_time = datetime.now()
        self.process_button.config(state=tk.DISABLED)
        self.select_button.config(state=tk.DISABLED)
        self.status_label.config(text="Détection des visages en cours...")

        def process():
            try:
                cap = cv2.VideoCapture(self.video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_count = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    if frame_count % 10 == 0:  # Traiter une frame sur 10
                        face_locations = self.collector.collect_faces(frame, frame_count)

                        if self.show_preview.get():
                            self.update_preview_with_faces(frame, face_locations)

                        self.update_progress((frame_count / total_frames) * 100)
                        self.update_stats(len(self.collector.detected_faces))

                cap.release()
                self.root.after(0, self.show_face_selection)

            except Exception as e:
                self.root.after(0, lambda: self.processing_error(str(e)))

        threading.Thread(target=process, daemon=True).start()

    def update_preview_with_faces(self, frame, face_locations):
        for top, right, bottom, left in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        height, width = frame.shape[:2]
        max_size = 400
        if height > max_size or width > max_size:
            scale = max_size / max(height, width)
            frame = cv2.resize(frame, None, fx=scale, fy=scale)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)
        self.preview_label.config(image=img_tk)
        self.preview_label.image = img_tk

    def show_face_selection(self):
        if not self.collector.detected_faces:
            messagebox.showinfo("Information", "Aucun visage détecté dans la vidéo")
            self.processing_completed()
            return

        selection_window = tk.Toplevel(self.root)
        selection_window.title("Sélection des visages")
        selection_window.geometry("800x600")

        # Frame principal avec scrollbar
        main_frame = ttk.Frame(selection_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Canvas et scrollbar
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Configuration du layout
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Grille de visages
        for i, face in enumerate(self.collector.detected_faces):
            frame = ttk.LabelFrame(scrollable_frame, text=f"Visage {i + 1}")
            frame.grid(row=i // 3, column=i % 3, padx=5, pady=5, sticky="nsew")

            # Convertir et afficher l'image
            img = Image.fromarray(cv2.cvtColor(face.image, cv2.COLOR_BGR2RGB))
            if img.size[0] > 150 or img.size[1] > 150:
                img.thumbnail((150, 150), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)

            label = ttk.Label(frame, image=img_tk)
            label.image = img_tk  # Garder une référence!
            label.pack(padx=5, pady=5)

            # Informations sur le visage
            ttk.Label(frame, text=f"Qualité: {face.quality:.2f}").pack()
            ttk.Label(frame, text=f"Frame: {face.frame_number}").pack()

        # Bouton de finalisation
        ttk.Button(
            selection_window,
            text="Terminer la sélection",
            command=lambda: self.finalize_selection(selection_window)
        ).pack(pady=10)

    def update_progress(self, value):
        self.progress_var.set(value)
        elapsed_time = datetime.now() - self.start_time
        minutes = int(elapsed_time.total_seconds() // 60)
        seconds = int(elapsed_time.total_seconds() % 60)
        self.processing_time_label.config(
            text=f"Temps écoulé : {minutes:02d}:{seconds:02d}"
        )

    def update_stats(self, faces_count):
        self.faces_found_label.config(text=f"Visages détectés : {faces_count}")

    def processing_error(self, error_message):
        messagebox.showerror("Erreur de traitement", error_message)
        self.processing_completed()

    def processing_completed(self):
        self.processing = False
        self.process_button.config(state=tk.NORMAL)
        self.select_button.config(state=tk.NORMAL)
        self.status_label.config(text="Traitement terminé")

    def finalize_selection(self, selection_window):
        selection_window.destroy()
        self.processing_completed()

    def on_closing(self):
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    app = VideoImportInterface()
    app.root.mainloop()