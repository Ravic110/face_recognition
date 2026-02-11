# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
import numpy as np
import os
import face_recognition
import json
from datetime import datetime
from tkinter.filedialog import askopenfilename
from threading import Thread
from config import ENCODED_DIR
from encodings_store import load_encodings_map, delete_encoding as delete_stored_encoding
import import_image
from playsound import playsound  # Importer le module pour jouer un son
import pygame


def draw_bold_text(image, text, position, font, font_scale, color, thickness):
    # Dessiner le texte plusieurs fois avec un décalage pour simuler le gras
    x, y = position
    cv2.putText(image, text, (x - 1, y), font, font_scale, color, thickness)
    cv2.putText(image, text, (x + 1, y), font, font_scale, color, thickness)
    cv2.putText(image, text, (x, y - 1), font, font_scale, color, thickness)
    cv2.putText(image, text, (x, y + 1), font, font_scale, color, thickness)
    # Dessiner le texte principal par-dessus
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness + 1)


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconnaissance Faciale")

        self.cap = None  # Pour l'objet vidéo OpenCV
        self.running = False  # Flag pour arrêter la capture
        self.new_faces = []  # Stocker encodages de nouveaux visages capturés
        self.target_person = None  # Nom de la personne à traquer
        self.alarm_running = False  # Indique si l'alarme est en cours
        self.cached_encodings = None

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
        self.is_capturing = False  # Flag pour éviter les boucles infinies

        self.setup_ui()

    def verify_face(self, face_encoding, threshold=0.5):
        """
        Vérifier si le visage existe avec une meilleure précision.
        :param face_encoding: L'encodage du visage à vérifier.
        :param threshold: Le seuil de distance pour considérer un visage comme correspondant.
        """
        encoded_faces = self.load_all_encodings()
        best_match = None
        min_distance = float('inf')

        for name, known_encoding in encoded_faces.items():
            face_distance = face_recognition.face_distance([known_encoding], face_encoding)[0]
            if face_distance < threshold and face_distance < min_distance:
                min_distance = face_distance
                best_match = name

        return (best_match is not None), best_match

    def load_all_encodings(self):
        return load_encodings_map()

    def get_cached_encodings(self):
        if self.cached_encodings is None:
            self.cached_encodings = self.load_all_encodings()
        return self.cached_encodings


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

        self.start_camera_button = ttk.Button(
            self.button_frame,
            text="Démarrer la caméra",
            command=self.start_camera,
            width=20
        )
        self.start_camera_button.pack(side="left", padx=5)

        self.capture_button = ttk.Button(
            self.button_frame,
            text="Capturer",
            command=self.capture_faces,
            width=20,
            state="disabled"
        )
        self.capture_button.pack(side="left", padx=5)

        self.stop_camera_button = ttk.Button(
            self.button_frame,
            text="Arrêter la caméra",
            command=self.stop_camera,
            width=20,
            state="disabled"
        )
        self.stop_camera_button.pack(side="left", padx=5)

        self.manage_encodings_button = ttk.Button(
            self.button_frame,
            text="Gérer les encodages",
            command=self.open_manage_window,
            width=20
        )
        self.manage_encodings_button.pack(side="left", padx=5)

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
        self.face_selector = ttk.Combobox(
            self.selection_frame,
            textvariable=self.face_var,
            state="readonly",
            values=[]  # Initialement vide, mis à jour après détection
        )
        self.face_selector.pack(side="left", padx=5)

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

        # Frame pour la fonctionnalité "alerté"
        self.alert_frame = ttk.LabelFrame(
            self.main_frame,
            text="Alertes",
            padding="10"
        )
        self.alert_frame.grid(row=5, column=0, sticky="ew", pady=10)

        ttk.Label(self.alert_frame, text="Personne à traquer :").pack(side="left", padx=5)
        self.target_var = tk.StringVar(value="")
        self.target_selector = ttk.Combobox(
            self.alert_frame,
            textvariable=self.target_var,
            state="readonly",
            values=[]  # Initialement vide, mis à jour après chargement des encodages
        )
        self.target_selector.pack(side="left", padx=5)

        self.set_target_button = ttk.Button(
            self.alert_frame,
            text="Définir comme cible",
            command=self.set_target_person
        )
        self.set_target_button.pack(side="left", padx=5)

        # Bouton pour arrêter l'alarme
        self.stop_alarm_button = ttk.Button(
            self.alert_frame,
            text="Arrêter l'alarme",
            command=self.stop_alarm,
            state="disabled"  # Désactivé par défaut
        )
        self.stop_alarm_button.pack(side="left", padx=5)

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
                    if known_name == self.target_person:
                        # Personne traquée - Rouge
                        cv2.rectangle(image_resized, (left, top), (right, bottom), (0, 0, 255), 2)
                        draw_bold_text(
                            image_resized, f"{known_name}", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
                        )
                    else:
                        # Visage connu - Vert
                        cv2.rectangle(image_resized, (left, top), (right, bottom), (0, 255, 0), 2)
                        draw_bold_text(
                            image_resized, f"{known_name}", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                        )
                else:
                    # Nouveau visage - Bleu
                    cv2.rectangle(image_resized, (left, top), (right, bottom), (255, 0, 0), 2)
                    draw_bold_text(
                        image_resized, f"inconnu {i + 1}", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
                    )

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
            self.face_selector.config(state="readonly")
            # Met à jour les options du menu déroulant avec les indices des visages
            self.face_selector["values"] = [f"Visage {i + 1}" for i in range(len(self.face_locations))]

            self.face_var.set("")  # Réinitialise la sélection
            self.select_button.config(state="normal")
        else:
            self.info_label.config(text="Aucun visage détecté")
            self.face_selector.config(state="disabled")
            self.face_var.set("")
            self.select_button.config(state="disabled")

    def select_face(self):
        try:
            face_label = self.face_var.get()  # Récupère la sélection comme "Visage X"
            if not face_label.startswith("Visage "):
                messagebox.showwarning("Attention", "Sélection invalide")
                return

            face_index = int(face_label.split(" ")[1]) - 1  # Extraire le numéro après "Visage"

            if 0 <= face_index < len(self.face_locations):
                self.selected_face_index = face_index
                exists, _ = self.verify_face(self.face_encodings[self.selected_face_index])

                if exists:
                    messagebox.showwarning("Attention", "Ce visage est déjà enregistré")
                    return

                self.save_button.config(state="normal")
                self.name_entry.focus()
            else:
                messagebox.showwarning("Attention", "Sélection invalide")
        except ValueError:
            messagebox.showwarning("Attention", "Sélection invalide")

    def save_face(self):
        if self.selected_face_index is None:
            return

        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Attention", "Veuillez entrer un nom")
            return

        if name in self.load_all_encodings():
            messagebox.showwarning("Attention", f"Le nom '{name}' existe déjà. Veuillez choisir un autre nom.")
            return

        try:
            import_image.save_face_encoding(name, self.face_encodings[self.selected_face_index])
            messagebox.showinfo("Succès", f"Visage de {name} enregistré avec succès")

            # Actualise la liste et l'affichage
            self.cached_encodings = None
            self.update_faces_after_save()

            # Réinitialise le champ de nom
            self.name_entry.delete(0, tk.END)
        except Exception as e:
            messagebox.showerror("Erreur", str(e))

    def update_faces_after_save(self):
        # Supprime le visage enregistré des listes
        if self.selected_face_index is not None:
            del self.face_locations[self.selected_face_index]
            del self.face_encodings[self.selected_face_index]

        # Réinitialise la sélection
        self.selected_face_index = None
        self.face_var.set("")
        self.save_button.config(state="disabled")

        # Actualise les options du menu déroulant
        if self.face_locations:
            self.face_selector["values"] = [f"Visage {i + 1}" for i in range(len(self.face_locations))]
        else:
            self.face_selector["values"] = []
            self.face_selector.config(state="disabled")

        # Actualise l'image affichée pour refléter les changements
        self.progress.start()
        Thread(target=self.process_image, daemon=True).start()

    def start_camera(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Impossible d'ouvrir la caméra")

            self.running = True
            self.capture_button.config(state="normal")
            self.start_camera_button.config(state="disabled")
            self.update_frame()
            self.stop_camera_button.config(state="normal")

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du démarrage de la caméra : {e}")

    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            print("Erreur : Impossible de lire la trame de la caméra.")
            return

        # Convertir en RGB pour traitement
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Réduire la taille pour accélérer la détection
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Détecter les visages
        face_locations = face_recognition.face_locations(small_frame)

        if face_locations:
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)
            known_encodings = self.get_cached_encodings()

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                recognized = False
                recognized_name = "Inconnu"

                for name, known_encoding in known_encodings.items():
                    face_distance = face_recognition.face_distance([known_encoding], face_encoding)[0]
                    if face_distance < 0.5:
                        recognized = True
                        recognized_name = name
                        break

                top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2

                if recognized and recognized_name == self.target_person:
                    color = (0, 0, 255)  # Rouge pour la personne traquée
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    draw_bold_text(frame, recognized_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    if not hasattr(self, "alarm_triggered") or not self.alarm_triggered:
                        self.alarm_triggered = True
                        Thread(target=self.play_alarm, daemon=True).start()
                        self.root.after(0, lambda: self.stop_alarm_button.config(state="normal"))

                elif recognized:
                    color = (0, 255, 0)  # Vert pour les visages connus
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    draw_bold_text(frame, recognized_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                else:
                    color = (255, 0, 0)  # Bleu pour les visages inconnus
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    draw_bold_text(frame, "Inconnu", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.current_image_tk = imgtk
        self.canvas.delete("all")
        self.canvas.create_image(self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2, anchor=tk.CENTER,
                                 image=imgtk)

        self.root.after(30, self.update_frame)

    def capture_faces(self):
        """Capture des visages depuis la vue caméra actuelle"""
        if not self.cap or not self.cap.isOpened():
            messagebox.showwarning("Attention", "La caméra n'est pas active")
            return

        self.capture_button.config(state="disabled")  # Désactiver le bouton pendant la capture

        try:
            # Prendre une image de la caméra
            ret, frame = self.cap.read()
            if not ret:
                raise Exception("Impossible de capturer l'image depuis la caméra")

            # Convertir en RGB pour traitement
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Réduire la taille pour accélérer la détection
            small_frame = cv2.resize(frame_rgb, (0, 0), fx=0.5, fy=0.5)

            # Détecter les visages
            face_locations = face_recognition.face_locations(small_frame)

            if not face_locations:
                messagebox.showinfo("Information", "Aucun visage détecté dans l'image")
                self.capture_button.config(state="normal")
                return

            # Encoder les visages détectés
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)

            # Préparation pour la capture dans une liste temporaire
            temp_faces = []

            # Pour chaque visage détecté
            for i, (face_location, face_encoding) in enumerate(zip(face_locations, face_encodings)):
                top, right, bottom, left = face_location

                # Vérifier si le visage existe déjà
                exists, known_name = self.verify_face(face_encoding)

                # Extraire le visage (à taille réelle)
                top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2  # Convertir à la taille réelle
                face_crop = frame_rgb[top:bottom, left:right]

                # Si le visage n'existe pas, l'ajouter à la liste temporaire
                if not exists:
                    temp_faces.append((face_encoding, face_crop, i + 1))

            # Si aucun nouveau visage n'a été trouvé
            if not temp_faces:
                messagebox.showinfo("Information", "Tous les visages détectés sont déjà enregistrés")
                self.capture_button.config(state="normal")
                return

            # Pour chaque nouveau visage, demander un nom
            for encoding, crop, face_num in temp_faces:
                # Afficher l'image du visage
                img = Image.fromarray(crop)
                img = img.resize((200, int(200 * img.height / img.width)))
                img_tk = ImageTk.PhotoImage(image=img)

                # Créer une fenêtre pour afficher le visage
                face_win = tk.Toplevel(self.root)
                face_win.title(f"Nouveau visage #{face_num}")
                face_win.transient(self.root)
                face_win.grab_set()

                # Afficher l'image
                img_label = ttk.Label(face_win, image=img_tk)
                img_label.image = img_tk  # Garder une référence
                img_label.pack(padx=20, pady=10)

                # Frame pour le nom
                name_frame = ttk.Frame(face_win)
                name_frame.pack(pady=10)

                ttk.Label(name_frame, text="Nom:").pack(side=tk.LEFT, padx=5)
                name_entry = ttk.Entry(name_frame)
                name_entry.pack(side=tk.LEFT, padx=5)
                name_entry.focus()

                # Variables pour stocker le résultat
                result_name = [None]  # Utiliser une liste pour permettre la modification dans la fonction interne

                def save_name():
                    name = name_entry.get().strip()
                    if name:
                        result_name[0] = name
                        face_win.destroy()
                    else:
                        messagebox.showwarning("Attention", "Veuillez entrer un nom", parent=face_win)

                def skip():
                    face_win.destroy()

                # Boutons
                btn_frame = ttk.Frame(face_win)
                btn_frame.pack(pady=10)

                ttk.Button(btn_frame, text="Enregistrer", command=save_name).pack(side=tk.LEFT, padx=5)
                ttk.Button(btn_frame, text="Ignorer", command=skip).pack(side=tk.LEFT, padx=5)

                # Centrer la fenêtre
                face_win.update_idletasks()
                w = face_win.winfo_width()
                h = face_win.winfo_height()
                x = (face_win.winfo_screenwidth() - w) // 2
                y = (face_win.winfo_screenheight() - h) // 2
                face_win.geometry(f"{w}x{h}+{x}+{y}")

                # Attendre que la fenêtre soit fermée
                self.root.wait_window(face_win)

                # Si un nom a été fourni, enregistrer le visage
                if result_name[0]:
                    try:
                        import_image.save_face_encoding(result_name[0], encoding)
                        messagebox.showinfo("Succès", f"Visage de {result_name[0]} enregistré avec succès")
                    except Exception as e:
                        messagebox.showerror("Erreur", str(e))

            # Mettre à jour la liste des cibles après l'enregistrement
            self.update_target_selector()

        except Exception as e:
            messagebox.showerror("Erreur", str(e))
        finally:
            self.capture_button.config(state="normal")  # Réactiver le bouton

    def stop_camera(self):
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None

        self.capture_button.config(state="disabled")
        self.stop_camera_button.config(state="disabled")
        self.start_camera_button.config(state="normal")

        # Nettoyage de l'image du canvas (optionnel)
        self.canvas.delete("all")

    def open_manage_window(self):
        manage_win = tk.Toplevel(self.root)
        manage_win.title("Gestion des visages encodés")
        manage_win.geometry("600x400")
        manage_win.transient(self.root)
        manage_win.grab_set()

        ttk.Label(manage_win, text="Visages enregistrés :", font=("Arial", 12)).pack(pady=10)

        listbox = tk.Listbox(manage_win)
        listbox.pack(fill="both", expand=True, padx=20)

        # On charge les encodages depuis metadata.json
        try:
            encodings = self.load_all_encodings()  # Ne pas modifier cette fonction
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur de chargement des encodages : {e}")
            return

        for name in sorted(encodings.keys()):
            listbox.insert(tk.END, name)

        def delete_selected():
            selected = listbox.curselection()
            if not selected:
                messagebox.showwarning("Attention", "Selectionnez un visage a supprimer.")
                return

            name = listbox.get(selected[0])
            confirm = messagebox.askyesno("Confirmation", f"Etes-vous sur de vouloir supprimer '{name}' ?")
            if not confirm:
                return

            try:
                removed = delete_stored_encoding(name)
                if not removed:
                    messagebox.showerror("Erreur", f"Le nom '{name}' n'existe pas.")
                    return

                listbox.delete(selected[0])
                messagebox.showinfo("Supprime", f"'{name}' a ete supprime.")
                self.refresh_encoding_list(listbox)
            except Exception as e:
                messagebox.showerror("Erreur", f"Une erreur est survenue lors de la suppression : {e}")

        ttk.Button(manage_win, text="Supprimer", command=delete_selected).pack(pady=10)

    def refresh_encoding_list(self, listbox):
        listbox.delete(0, tk.END)
        try:
            encodings = self.load_all_encodings()
            self.cached_encodings = encodings
            for name in sorted(encodings.keys()):
                listbox.insert(tk.END, name)
            self.update_target_selector()  # Mettre à jour le menu déroulant
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur de rechargement : {e}")

    def set_target_person(self):
        target_name = self.target_var.get()
        if not target_name:
            messagebox.showwarning("Attention", "Veuillez sélectionner une personne à traquer.")
            return  # Arrêter l'exécution ici

        self.target_person = target_name
        self.alarm_triggered = False  # Réinitialiser l'alarme
        messagebox.showinfo("Succès", f"'{target_name}' est maintenant la cible à traquer.")

        self.update_target_selector()

    def update_target_selector(self):
        """Met à jour le menu déroulant avec les noms des personnes enregistrées."""
        try:
            encodings = self.load_all_encodings()
            self.target_selector["values"] = sorted(encodings.keys())  # Ajouter les noms triés
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la mise à jour des cibles : {e}")

    def play_alarm(self):
        """Joue une alarme sonore en boucle jusqu'a ce qu'elle soit arretee."""
        alarm_path = os.path.join(os.path.dirname(__file__), "alarm", "alarm-301729.mp3")
        if not os.path.exists(alarm_path):
            self.root.after(0, lambda: messagebox.showerror("Erreur", f"Fichier audio introuvable : {alarm_path}"))
            return

        pygame.mixer.init()
        pygame.mixer.music.load(alarm_path)
        pygame.mixer.music.play(loops=-1)

        self.alarm_running = True
        self.root.after(0, lambda: self.stop_alarm_button.config(state="normal"))

    def stop_alarm(self):
        """Arrête l'alarme sonore immédiatement."""
        if self.alarm_running:
            pygame.mixer.music.stop()  # Arrêter la lecture du son
            self.alarm_running = False
            self.stop_alarm_button.config(state="disabled")  # Désactiver le bouton
            messagebox.showinfo("Alarme", "L'alarme a été arrêtée.")

def main():
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()