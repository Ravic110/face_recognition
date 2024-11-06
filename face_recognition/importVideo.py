import cv2
import face_recognition
from tkinter import Tk
from tkinter.filedialog import askopenfilename


def process_video(video_path):
    """
    Analyser la vidéo pour détecter les visages dans chaque frame.
    """
    # Charger la vidéo
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Erreur lors de l'ouverture de la vidéo.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Conversion de l'image en format RGB pour face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Détection des visages
        face_locations = face_recognition.face_locations(rgb_frame)

        # Dessiner des rectangles autour des visages détectés
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Afficher le résultat
        cv2.imshow("Détection de visages", frame)

        # Quitter la boucle si la touche 'q' est appuyée
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()


def import_video():
    """
    Ouvrir une boîte de dialogue pour sélectionner une vidéo.
    """
    Tk().withdraw()  # Fermer la fenêtre Tkinter par défaut
    video_path = askopenfilename(title="Sélectionnez une vidéo", filetypes=[("Vidéos", "*.mp4;*.avi;*.mkv")])

    if not video_path:
        print("Aucune vidéo sélectionnée.")
        return

    # Traiter la vidéo pour la détection de visages
    process_video(video_path)


if __name__ == "__main__":
    print("Détection de visages dans une vidéo")
    import_video()
