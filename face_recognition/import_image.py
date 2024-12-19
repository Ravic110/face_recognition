import hashlib

import cv2
import face_recognition
import os
import json
from datetime import datetime
from tkinter import Tk, messagebox
from tkinter.filedialog import askopenfilename
from config import ENCODED_DIR, META_FILE

def update_metadata(unique_id, name):
    metadata = {}
    if os.path.exists(META_FILE):
        with open(META_FILE, 'r') as f:
            metadata = json.load(f)

    metadata[unique_id] = {
        'name': name,
        'date_creation': datetime.now().isoformat()
    }

    with open(META_FILE, 'w') as f:
        json.dump(metadata, f, indent=4)

def save_face_encoding(name, face_encoding):
    """
    Enregistrer l'encodage du visage détecté dans un fichier JSON avec mise à jour du metadata.
    """
    # Création d'un identifiant unique
    timestamp = datetime.now().isoformat()
    unique_id = hashlib.sha256(f"{name}{timestamp}".encode()).hexdigest()[:12]

    # Formatage des données pour l'encodage
    encoding_data = {
        'name': name,
        'encoding': face_encoding.tolist(),
        'timestamp': timestamp
    }

    file_path = os.path.join(ENCODED_DIR, f"{unique_id}.json")
    with open(file_path, 'w') as file:
        json.dump(encoding_data, file)

    update_metadata(unique_id, name)
    print(f"Encodage pour {name} sauvegardé avec succès.")

def measure_sharpness(image):
    """
    Mesurer la netteté de l'image à l'aide de la variance du Laplacien.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

def calculate_face_precision(face_location, image_shape):
    """
    Calculer la précision de la détection du visage en fonction de sa taille par rapport à l'image.
    """
    top, right, bottom, left = face_location
    face_width = right - left
    face_height = bottom - top
    face_area = face_width * face_height

    image_area = image_shape[0] * image_shape[1]

    # Calculer la précision comme le pourcentage de l'aire du visage par rapport à l'aire de l'image
    precision_percentage = (face_area / image_area) * 100
    return precision_percentage

def select_face_from_image(name, image):
    """
    Permettre à l'utilisateur de sélectionner un visage avec une meilleure validation.
    """
    if image is None:
        print("Erreur : image corrompue ou impossible à lire.")
        return

    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    if not face_encodings:
        print("Aucun visage détecté. Veuillez réessayer avec une autre image.")
        return

    sharpness = measure_sharpness(image)
    print(f"Netteté de l'image: {sharpness:.2f}")

    # Dessiner des rectangles autour des visages détectés
    for i, (top, right, bottom, left) in enumerate(face_locations):
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, f"Visage {i + 1}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        precision = calculate_face_precision((top, right, bottom, left), image.shape)
        print(f"Précision de la détection du visage {i + 1}: {precision:.2f}%")

    # Afficher l'image avec les visages détectés
    cv2.imshow("Visages détectés", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Demander à l'utilisateur de sélectionner le visage à enregistrer
    while True:
        try:
            choice = int(input(f"Sélectionnez le numéro du visage à enregistrer (1-{len(face_encodings)}) : ")) - 1
            if 0 <= choice < len(face_encodings):
                save_face_encoding(name, face_encodings[choice])
                break
            else:
                print("Choix invalide. Veuillez réessayer.")
        except ValueError:
            print("Entrée non valide. Veuillez entrer un numéro valide.")

def import_face_from_image(name):
    """
    Importer une image depuis le système de fichiers et enregistrer l'encodage du visage sélectionné.
    """
    Tk().withdraw()  # Masquer la fenêtre Tkinter
    image_path = askopenfilename(title="Sélectionnez une image", filetypes=[("Images", "*.jpg;*.jpeg;*.png")])

    if not image_path:
        print("Aucune image sélectionnée.")
        return

    # Chargement de l'image
    image = cv2.imread(image_path)
    if image is None:
        print("Erreur lors du chargement de l'image. Le fichier peut être corrompu ou introuvable.")
        return

    # Vérification de la taille de l'image
    if image.size == 0:
        print("L'image est vide ou invalide.")
        return

    try:
        # Conversion en RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except cv2.error as e:
        print("Erreur lors de la conversion de l'image en RGB :", e)
        return

    # Sélection et encodage du visage
    select_face_from_image(name, rgb_image)

def capture_face(name):
    """
    Capturer une image depuis la webcam, détecter les visages et sauvegarder l'encodage d'un visage.
    """
    cap = cv2.VideoCapture(0)
    print("Appuyez sur la barre d'espace pour capturer une image, ou sur 'q' pour quitter.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur lors de la capture depuis la webcam.")
            break

        cv2.imshow("Webcam", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Barre d'espace pour capturer
            print("Image capturée.")
            cap.release()
            cv2.destroyAllWindows()
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except cv2.error as e:
                print("Erreur lors de la conversion de l'image en RGB :", e)
                return
            select_face_from_image(name, rgb_frame)
            break
        elif key == ord('q'):  # Quitter
            print("Capture annulée.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("1. Capture depuis la webcam")
    print("2. Importer une image")
    choice = input("Choisissez une option (1 ou 2) : ")

    name = input("Entrez le nom de la personne : ")

    if choice == '1':
        capture_face(name)
    elif choice == '2':
        import_face_from_image(name)
    else:
        print("Option invalide.")
