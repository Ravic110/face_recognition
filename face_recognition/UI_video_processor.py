import face_recognition
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox, Querybox
from PIL import Image, ImageTk
from tkinter import LEFT, Canvas, Frame, Scrollbar, filedialog
from video_processor import process_chunk
from utils import load_existing_encodings, save_face_encoding, delete_encoding, validate_encoding, is_duplicate
from config import ENCODED_DIR
import threading
import cv2
import os
import logging
import json
import numpy as np
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from queue import Queue

logging.basicConfig(filename='app.log', level=logging.INFO)


class VideoImporterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Processor Pro")
        self.root.geometry("1000x900")
        self.style = ttk.Style("solar")

        self.video_path = None
        self.tolerance = 0.5
        self.executor = ProcessPoolExecutor(max_workers=4)
        self.io_executor = ThreadPoolExecutor(max_workers=4)  # Pour les tâches I/O bound
        self.cpu_executor = ProcessPoolExecutor(max_workers=4)
        self.progress_queue = Queue()
        self.running = True
        self.existing_encodings = load_existing_encodings()

        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.check_progress()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Control Panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill='x', pady=10)

        ttk.Button(control_frame,
                   text="Manage Encodings",
                   command=self.show_encoding_manager,
                   bootstyle=INFO).pack(side=RIGHT, padx=5)

        ttk.Button(control_frame,
                   text="Select Video",
                   command=self.select_video,
                   bootstyle=PRIMARY).pack(side=LEFT, padx=5)

        # Ajout d'un Spinbox pour frame_skip
        frame_skip_frame = ttk.Frame(control_frame)
        frame_skip_frame.pack(side=LEFT, padx=10)

        ttk.Label(frame_skip_frame, text="Frame Skip:").pack(side=LEFT)
        self.frame_skip_var = ttk.IntVar(value=10)  # Valeur par défaut
        ttk.Spinbox(frame_skip_frame, from_=1, to=30, textvariable=self.frame_skip_var, width=5).pack(side=LEFT)

        # Preview
        self.preview_frame = ttk.LabelFrame(main_frame, text="Smart Preview")
        self.preview_frame.pack(fill='x', pady=10)

        self.preview_labels = [ttk.Label(self.preview_frame) for _ in range(3)]
        for label in self.preview_labels:
            label.pack(side=LEFT, padx=5)

        # Info
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill='x', pady=10)

        self.video_label = ttk.Label(info_frame, text="No video selected", width=60)
        self.video_label.pack(side=LEFT)

        self.frame_skip_info = ttk.Label(info_frame, text="Frame skip: -", bootstyle=INFO)
        self.frame_skip_info.pack(side=RIGHT)

        # Progress
        self.setup_progress_bars(main_frame)

        # Log
        self.log_text = ttk.Text(main_frame, height=8, wrap='word')
        self.log_text.pack(fill='x', pady=10)

    def setup_progress_bars(self, parent):
        progress_frame = ttk.LabelFrame(parent, text="Processing Stages")
        progress_frame.pack(fill='x', pady=10)

        self.stage_bars = []
        stages = [
            ("Video Analysis", "Scene detection"),
            ("Face Processing", "Multi-core processing"),
            ("Result Merging", "Database comparison")
        ]

        for idx, (title, desc) in enumerate(stages):
            frame = ttk.Frame(progress_frame)
            frame.pack(fill='x', pady=5)

            ttk.Label(frame, text=f"{idx + 1}. {title}", width=25).pack(side=LEFT)
            bar = ttk.Progressbar(frame, orient=HORIZONTAL, length=300, mode="determinate")
            bar.pack(side=LEFT, padx=5)
            ttk.Label(frame, text=desc, width=30).pack(side=LEFT)

            self.stage_bars.append(bar)

    def select_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Videos", "*.mp4;*.avi;*.mov")])
        if self.video_path:
            self.video_label.config(text=os.path.basename(self.video_path))
            self.generate_smart_preview()
            self.start_processing()

    def generate_smart_preview(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.log_error("Erreur : Impossible d'ouvrir la vidéo.")
                return

            key_frames = []
            frame_skip = self.frame_skip_var.get()  # Récupérer la valeur de frame_skip
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Si la vidéo est trop courte, ajuster frame_skip
            if total_frames < 100 * frame_skip:
                frame_skip = max(1, total_frames // 100)  # Ne pas descendre en dessous de 1

            # Lire la première frame
            ret, prev_frame = cap.read()
            if not ret:
                self.log_error("Erreur : Impossible de lire la première frame.")
                return

            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

            # Parcourir la vidéo avec frame_skip
            for i in range(1, 100):
                frame_id = i * 100 * frame_skip  # Calculer l'ID de la frame à lire
                if frame_id >= total_frames:
                    break  # Sortir si on dépasse le nombre total de frames

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = cap.read()
                if not ret:
                    break

                # Calculer la différence entre les frames
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                delta = cv2.absdiff(gray, prev_gray)
                thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
                change = cv2.countNonZero(thresh)

                # Si le changement est significatif, ajouter la frame à key_frames
                if change > 5000:
                    key_frames.append((change, frame))

                prev_gray = gray

            # Trier les frames par ordre de changement décroissant
            key_frames.sort(reverse=True, key=lambda x: x[0])

            # Afficher les 3 meilleures frames dans les labels de prévisualisation
            for idx, (_, frame) in enumerate(key_frames[:3]):
                self.update_preview_label(idx, frame)

        except Exception as e:
            self.log_error(f"Erreur lors de la génération de la prévisualisation : {str(e)}")
        finally:
            if 'cap' in locals():
                cap.release()

    def update_preview_label(self, idx, frame):
        try:
            frame = cv2.resize(frame, (300, 200))  # Redimensionner la frame
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertir en RGB
            img = ImageTk.PhotoImage(Image.fromarray(img))  # Convertir en format Tkinter
            self.preview_labels[idx].config(image=img)
            self.preview_labels[idx].image = img  # Garder une référence pour éviter la garbage collection
        except Exception as e:
            self.log_error(f"Erreur lors de la mise à jour de la prévisualisation : {str(e)}")

    def start_processing(self):
        threading.Thread(target=self.process_video_parallel, daemon=True).start()

    def process_video_parallel(self):
        try:
            chunks = self.prepare_video_chunks()
            futures = []
            frame_skip = self.frame_skip_var.get()

            for i, chunk in enumerate(chunks):
                future = self.executor.submit(process_chunk, (*chunk, i, frame_skip))  # Ajout de frame_skip
                futures.append((future, i))
                self.update_progress(0, (i / len(chunks)) * 100)

            temp_files = []
            for future, chunk_id in futures:
                try:
                    temp_file = future.result()
                    temp_files.append(temp_file)
                    self.update_progress(1, (chunk_id / len(chunks)) * 100)
                except Exception as e:
                    self.log_error(f"Chunk {chunk_id} error: {str(e)}")

            all_faces = self.merge_temp_files(temp_files)
            self.process_final_results(all_faces)

        except Exception as e:
            self.log_error(f"Processing error: {str(e)}")

    def prepare_video_chunks(self):
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        chunk_size = total_frames // 4
        return [
            (self.video_path, i * chunk_size, (i + 1) * chunk_size)
            for i in range(4)
        ]

    def merge_temp_files(self, temp_files):
        all_faces = []
        for temp_file in temp_files:
            try:
                with open(temp_file, 'r') as f:
                    chunk_data = json.load(f)
                    for face in chunk_data:
                        face['image'] = bytes.fromhex(face['image'])  # Convertir l'hexadécimal en bytes
                        face['encoding'] = np.array(face['encoding'])
                    all_faces.extend(chunk_data)
                os.remove(temp_file)
            except Exception as e:
                self.log_error(f"Merge error: {str(e)}")
        return all_faces

    def process_final_results(self, all_faces):
        self.unique_faces = self.group_similar_faces(all_faces)
        self.root.after(0, self.show_faces_for_selection, self.unique_faces)
        self.update_progress(2, 100)

    def group_similar_faces(self, faces):
        """Regroupe les visages similaires en utilisant une double vérification"""
        unique_groups = []
        total_faces = len(faces)

        for idx, face_data in enumerate(faces):
            try:
                # Conversion des données sérialisées
                face_image = np.frombuffer(face_data['image'], dtype=np.uint8)
                face_image = cv2.imdecode(face_image, cv2.IMREAD_COLOR)
                face_encoding = np.array(face_data['encoding'])

                # Mise à jour de la progression
                self.update_progress(2, (idx / total_faces) * 100)

                # Vérification des doublons existants
                existing_name = is_duplicate(face_encoding, self.existing_encodings, self.tolerance)
                if existing_name:
                    unique_groups.append({
                        'name': existing_name,
                        'count': 1,
                        'encoding': face_encoding,
                        'thumbnail': face_image
                    })
                    continue

                # Vérification des similarités dans les nouveaux visages
                is_new_group = True
                for group in unique_groups:
                    if self.are_faces_similar(face_encoding, group['encoding']):
                        group['count'] += 1
                        if self.get_image_sharpness(face_image) > self.get_image_sharpness(group['thumbnail']):
                            group['thumbnail'] = face_image
                        is_new_group = False
                        break

                if is_new_group:
                    unique_groups.append({
                        'name': None,
                        'count': 1,
                        'encoding': face_encoding,
                        'thumbnail': face_image
                    })

            except Exception as e:
                self.log_error(f"Erreur traitement visage {idx}: {str(e)}")

        return unique_groups

    def are_faces_similar(self, encoding1, encoding2):
        """Compare deux encodages faciaux pour déterminer s'ils sont similaires"""
        return face_recognition.compare_faces([encoding1], encoding2, tolerance=self.tolerance)[0]

    def get_image_sharpness(self, image):
        """Calcule la netteté d'une image avec le laplacien"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def show_faces_for_selection(self, face_groups):
        """Affiche les groupes de visages détectés avec possibilité de sélection"""
        if hasattr(self, 'current_face_window') and self.current_face_window.winfo_exists():
            self.current_face_window.destroy()

        selection_window = ttk.Toplevel(self.root)
        self.current_face_window = selection_window
        selection_window.title("Groupes de Visages - Sélection")
        selection_window.geometry("1000x800")

        # Configuration du défilement
        canvas = Canvas(selection_window, bg="white")
        scrollbar = ttk.Scrollbar(selection_window, orient="vertical", command=canvas.yview)
        content_frame = ttk.Frame(canvas)

        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.create_window((0, 0), window=content_frame, anchor="nw")

        # Affichage des groupes
        for idx, group in enumerate(face_groups):
            group_frame = ttk.Frame(content_frame)
            group_frame.pack(fill="x", pady=10, padx=20)

            # Miniature + informations
            thumbnail = self.get_tk_thumbnail(group['thumbnail'])
            ttk.Label(group_frame, image=thumbnail).image = thumbnail
            ttk.Label(group_frame, image=thumbnail).pack(side="left")

            info_frame = ttk.Frame(group_frame)
            info_frame.pack(side="left", padx=10, fill="x", expand=True)

            # Nom existant ou compteur
            if group['name']:
                ttk.Label(info_frame,
                          text=f"Personne existante : {group['name']}",
                          font=("Helvetica", 12, "bold"),
                          bootstyle=SUCCESS).pack(anchor="w")
            else:
                ttk.Label(info_frame,
                          text=f"Apparitions similaires : {group['count']}",
                          font=("Helvetica", 10),
                          bootstyle=INFO).pack(anchor="w")

                # Bouton d'assignation
                ttk.Button(info_frame,
                           text="Nommer ce groupe",
                           command=lambda g=group: self.prompt_for_group_name(g),
                           bootstyle=(OUTLINE, PRIMARY)).pack(pady=5)

        # Gestion du redimensionnement
        content_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # Affichage des statistiques
        stats_frame = ttk.Frame(selection_window)
        stats_frame.pack(pady=10)
        ttk.Label(stats_frame,
                  text=f"{len(face_groups)} groupes uniques détectés",
                  bootstyle=INFO).pack()

    def get_tk_thumbnail(self, cv_image):
        """Convertit une image OpenCV en format Tkinter"""
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image).resize((150, 150), Image.LANCZOS)
        return ImageTk.PhotoImage(pil_image)

    def prompt_for_group_name(self, group):
        name = Querybox.get_string("Nom du groupe",
                                   f"Entrez le nom pour ces {group['count']} apparitions :",
                                   parent=self.current_face_window)
        if name and isinstance(name, str) and name.strip():
            save_face_encoding(name.strip(), group['encoding'], group['thumbnail'])
            self.existing_encodings = load_existing_encodings()
            self.refresh_face_window()
        else:
            Messagebox.show_warning("Nom invalide", "Veuillez entrer un nom valide.")

    def refresh_face_window(self):
        if not hasattr(self, 'unique_faces') or not self.unique_faces:
            return
        if hasattr(self, 'current_face_window') and self.current_face_window.winfo_exists():
            self.current_face_window.destroy()
            self.show_faces_for_selection(self.unique_faces)

    def show_encoding_manager(self):
        if hasattr(self, 'encoding_manager_window') and self.encoding_manager_window.winfo_exists():
            self.encoding_manager_window.lift()
            return
        manager = ttk.Toplevel(self.root)
        manager.title("Encoding Manager")
        manager.geometry("800x600")

        self.encoding_progress = ttk.Progressbar(manager, orient=HORIZONTAL, length=300, mode="determinate")
        self.encoding_progress.pack(pady=10)
        self.io_executor.submit(self.load_encodings_async, manager)

    def load_encodings_async(self, window):
        encodings = load_existing_encodings()
        self.root.after(0, self.display_encodings, window, encodings)

    def display_encodings(self, window, encodings):
        # Effacer le contenu actuel de la fenêtre
        for widget in window.winfo_children():
            widget.destroy()

        # Recréer le canvas et la scrollbar
        canvas = Canvas(window)
        scrollbar = ttk.Scrollbar(window, orient="vertical", command=canvas.yview)
        frame = Frame(canvas)

        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.create_window((0, 0), window=frame, anchor="nw")

        # Afficher les encodings
        for enc in encodings:
            row = ttk.Frame(frame)
            row.pack(fill='x', pady=5)

            img_data = self.load_encoding_data(enc['name'])
            if img_data:
                img = self.convert_cv_to_tk(img_data)
                ttk.Label(row, image=img).pack(side=LEFT)

            ttk.Label(row, text=enc['name'], width=30).pack(side=LEFT)
            ttk.Button(row,
                       text="Delete",
                       command=lambda n=enc['name']: self.delete_encoding(n, window),
                       bootstyle=DANGER).pack(side=RIGHT)

        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    def load_encoding_data(self, name):
        for filename in os.listdir(ENCODED_DIR):
            if filename.endswith(".json") and name in filename:
                with open(os.path.join(ENCODED_DIR, filename), 'r') as f:
                    data = json.load(f)
                    return data.get('image')
        return None

    def convert_cv_to_tk(self, img_data):
        try:
            img_array = np.array(img_data, dtype=np.uint8)
            img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img).resize((100, 100))
            return ImageTk.PhotoImage(img)
        except Exception as e:
            self.log_error(f"Image conversion error: {str(e)}")
            return None

    def delete_encoding(self, name, window):
        if Messagebox.show_question(f"Delete {name}?", parent=window):
            delete_encoding(name)
            self.existing_encodings = load_existing_encodings()
            self.display_encodings(window, self.existing_encodings)  # Mettre à jour la fenêtre

    def update_progress(self, stage_idx, progress):
        self.stage_bars[stage_idx]['value'] = progress
        self.root.update_idletasks()

    def check_progress(self):
        while not self.progress_queue.empty():
            chunk_id, progress = self.progress_queue.get()
            self.update_progress(1, progress)
        if self.running:
            self.root.after(100, self.check_progress)

    def log_error(self, message):
        self.log_text.insert('end', f"[ERROR] {message}\n")
        logging.error(message)

    def on_closing(self):
        self.running = False
        self.executor.shutdown(wait=False)
        self.root.destroy()


if __name__ == "__main__":
    root = ttk.Window(themename="solar")
    app = VideoImporterApp(root)
    root.mainloop()