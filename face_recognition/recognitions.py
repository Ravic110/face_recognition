import cv2
import face_recognition
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Dossier pour stocker les visages encodés
ENCODED_DIR = 'encodings'

if not os.path.exists(ENCODED_DIR):
    os.makedirs(ENCODED_DIR)


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


def save_face_encoding(name, face_encoding):
    """
    Enregistrer l'encodage du visage détecté dans un fichier.
    """
    file_path = f"{ENCODED_DIR}/{name}.txt"
    if os.path.exists(file_path):
        print(f"Un fichier pour {name} existe déjà. Veuillez utiliser un autre nom.")
        return

    # Sauvegarder l'encodage dans un fichier
    with open(file_path, 'w') as file:
        file.write(str(face_encoding.tolist()))
    print(f"Encodage pour {name} sauvegardé avec succès.")


def select_face_from_image(name, image):
    """
    Permettre à l'utilisateur de sélectionner un visage dans une image où plusieurs visages sont détectés.
    """
    # Conversion de l'image en format RGB pour face_recognition
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Détection des visages et encodage
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    if not face_encodings:
        print("Aucun visage détecté. Veuillez réessayer avec une autre image.")
        return

    # Mesurer la netteté de l'image
    sharpness = measure_sharpness(image)
    print(f"Netteté de l'image: {sharpness:.2f}")

    # Afficher chaque visage détecté
    for i, (top, right, bottom, left) in enumerate(face_locations):
        # Extraire le visage
        face_image = image[top:bottom, left:right]
        # Redimensionner pour un meilleur affichage
        face_image = cv2.resize(face_image, (150, 150))
        cv2.imshow(f"Visage {i + 1}", face_image)

        # Calculer la précision de la détection du visage
        precision = calculate_face_precision((top, right, bottom, left), image.shape)
        print(f"Précision de la détection du visage {i + 1}: {precision:.2f}%")

        cv2.waitKey(500)  # Attendre 500ms pour que les fenêtres soient visibles

    # Demander à l'utilisateur de choisir un visage
    choice = int(input(f"Sélectionnez le numéro du visage à enregistrer (1-{len(face_encodings)}) : ")) - 1

    if 0 <= choice < len(face_encodings):
        # Sauvegarder l'encodage du visage sélectionné
        save_face_encoding(name, face_encodings[choice])
    else:
        print("Choix invalide. Aucune action effectuée.")

    # Fermer les fenêtres d'affichage des visages
    cv2.destroyAllWindows()


def import_face_from_image(name):
    """
    Importer une image depuis le système de fichiers et enregistrer l'encodage du visage sélectionné.
    """
    Tk().withdraw()
    image_path = askopenfilename(title="Sélectionnez une image", filetypes=[("Images", "*.jpg;*.jpeg;*.png")])

    if not image_path:
        print("Aucune image sélectionnée.")
        return

    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        print("Erreur lors du chargement de l'image.")
        return

    # Sélectionner un visage dans l'image
    select_face_from_image(name, image)


def capture_face(name):
    pass


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
