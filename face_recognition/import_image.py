import cv2
import face_recognition
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

from encodings_store import save_face_encoding as store_save_face_encoding


def save_face_encoding(name, face_encoding):
    """
    Enregistrer l'encodage du visage detecte dans un fichier JSON.
    """
    try:
        store_save_face_encoding(name, face_encoding)
        print(f"Encodage pour {name} sauvegarde avec succes.")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde de l'encodage : {e}")


def measure_sharpness(image):
    """
    Mesurer la nettete de l'image a l'aide de la variance du Laplacien.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var


def calculate_face_precision(face_location, image_shape):
    """
    Calculer la precision de la detection du visage en fonction de sa taille par rapport a l'image.
    """
    top, right, bottom, left = face_location
    face_width = right - left
    face_height = bottom - top
    face_area = face_width * face_height

    image_area = image_shape[0] * image_shape[1]
    precision_percentage = (face_area / image_area) * 100
    return precision_percentage


def select_face_from_image(image):
    if image is None:
        print("Erreur : image corrompue ou impossible a lire.")
        return None, None

    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    if not face_encodings:
        print("Aucun visage detecte. Veuillez reessayer avec une autre image.")
        return None, None

    sharpness = measure_sharpness(image)
    print(f"Nettete de l'image: {sharpness:.2f}")

    resized_image = resize_image_to_fit_screen(image)

    scale_factor = calculate_text_scale(resized_image.shape)
    text_scale = 0.8 * scale_factor
    line_thickness = int(2 * scale_factor)

    for i, (top, right, bottom, left) in enumerate(face_locations):
        top = int(top * resized_image.shape[0] / image.shape[0])
        right = int(right * resized_image.shape[1] / image.shape[1])
        bottom = int(bottom * resized_image.shape[0] / image.shape[0])
        left = int(left * resized_image.shape[1] / image.shape[1])

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
        print(f"Precision de la detection du visage {i + 1}: {precision:.2f}%")

    resized_image_bgr = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)

    cv2.imshow("Visages detectes", resized_image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    while True:
        try:
            choice = int(input(f"Selectionnez le numero du visage a enregistrer (1-{len(face_encodings)}) : ")) - 1
            if 0 <= choice < len(face_encodings):
                return face_encodings[choice], sharpness
            print("Choix invalide. Veuillez reessayer.")
        except ValueError:
            print("Entree non valide. Veuillez entrer un numero valide.")


def import_face_from_image():
    """
    Importer une image depuis le systeme de fichiers, selectionner un visage et demander un nom.
    """
    Tk().withdraw()
    image_path = askopenfilename(title="Selectionnez une image", filetypes=[("Images", "*.jpg;*.jpeg;*.png")])

    if not image_path:
        print("Aucune image selectionnee.")
        return

    image = cv2.imread(image_path)
    if image is None:
        print("Erreur lors du chargement de l'image. Le fichier peut etre corrompu ou introuvable.")
        return

    if image.size == 0:
        print("L'image est vide ou invalide.")
        return

    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except cv2.error as e:
        print("Erreur lors de la conversion de l'image en RGB :", e)
        return

    face_encoding, _ = select_face_from_image(rgb_image)

    if face_encoding is not None:
        name = input("Entrez le nom de la personne pour le visage selectionne : ")
        save_face_encoding(name, face_encoding)


def resize_image_to_fit_screen(image, max_width=1280, max_height=720):
    """
    Redimensionner une image pour qu'elle s'adapte a l'ecran tout en conservant le ratio.
    """
    height, width = image.shape[:2]

    width_ratio = max_width / width
    height_ratio = max_height / height
    scale_factor = min(width_ratio, height_ratio, 1.0)

    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image


def calculate_text_scale(image_shape, base_width=800):
    """
    Calcule une echelle pour le texte et l'epaisseur des lignes en fonction de la largeur de l'image.
    """
    current_width = image_shape[1]
    scale_factor = current_width / base_width
    return max(scale_factor, 0.5)


if __name__ == "__main__":
    import_face_from_image()
