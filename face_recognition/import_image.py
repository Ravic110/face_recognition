import hashlib
import cv2
import face_recognition
import os
import json
from datetime import datetime
from tkinter import Tk
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

    # Redimensionner l'image pour l'adapter à l'écran
    resized_image = resize_image_to_fit_screen(image)

    # Calculer l'échelle dynamique pour le texte et les lignes
    scale_factor = calculate_text_scale(resized_image.shape)
    text_scale = 0.8 * scale_factor
    line_thickness = int(2 * scale_factor)

    # Dessiner des rectangles et ajouter du texte
    for i, (top, right, bottom, left) in enumerate(face_locations):
        # Adapter les coordonnées pour l'image redimensionnée
        top = int(top * resized_image.shape[0] / image.shape[0])
        right = int(right * resized_image.shape[1] / image.shape[1])
        bottom = int(bottom * resized_image.shape[0] / image.shape[0])
        left = int(left * resized_image.shape[1] / image.shape[1])

        # Dessiner les rectangles et textes
        cv2.rectangle(resized_image, (left, top), (right, bottom), (0, 255, 0), line_thickness)
        cv2.putText(
            resized_image,
            f"Visage {i + 1}",
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (0, 255, 0),
            max(1, line_thickness // 2)
        )

        precision = calculate_face_precision((top, right, bottom, left), image.shape)
        print(f"Précision de la détection du visage {i + 1}: {precision:.2f}%")

    # Si nécessaire, reconvertir en BGR avant l'affichage
    resized_image_bgr = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)

    # Afficher l'image avec les visages détectés
    cv2.imshow("Visages détectés", resized_image_bgr)
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


def resize_image_to_fit_screen(image, max_width=1280, max_height=720):
    """
    Redimensionner une image pour qu'elle s'adapte à l'écran tout en conservant le ratio.
    :param image: L'image d'entrée (numpy array).
    :param max_width: Largeur maximale autorisée.
    :param max_height: Hauteur maximale autorisée.
    :return: Image redimensionnée.
    """
    height, width = image.shape[:2]

    # Calculer les ratios de redimensionnement pour largeur et hauteur
    width_ratio = max_width / width
    height_ratio = max_height / height

    # Prendre le ratio minimum pour conserver le ratio d'aspect
    scale_factor = min(width_ratio, height_ratio, 1.0)  # Ne pas agrandir si l'image est plus petite que l'écran

    # Calculer la nouvelle taille
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Redimensionner l'image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image


def calculate_text_scale(image_shape, base_width=800):
    """
    Calcule une échelle pour le texte et l'épaisseur des lignes en fonction de la largeur de l'image.
    :param image_shape: Dimensions de l'image (hauteur, largeur).
    :param base_width: Largeur de référence pour une taille de texte standard.
    :return: Un facteur d'échelle pour ajuster la taille du texte et l'épaisseur des lignes.
    """
    current_width = image_shape[1]
    scale_factor = current_width / base_width
    return max(scale_factor, 0.5)  # Ne pas trop réduire pour des images très petites


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
    name = input("Entrez le nom de la personne : ")
    import_face_from_image(name)
