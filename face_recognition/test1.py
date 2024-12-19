import face_recognition
import cv2
import os

# Chemin vers le dossier où stocker les images des visages
known_faces_dir = 'known_faces'
if not os.path.exists(known_faces_dir):
    os.makedirs(known_faces_dir)

# Charger les images des visages connus et leurs noms
known_face_encodings = []
known_face_names = []
for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image = face_recognition.load_image_file(f"{known_faces_dir}/{filename}")
        face_encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(filename[:-4])

# Initialiser la vidéo
video_capture = cv2.VideoCapture(0)

# Boucle infinie pour traiter les images de la vidéo
while True:
    # Capturer une image de la vidéo
    ret, frame = video_capture.read()

    # Convertir l'image en RGB (nécessaire pour face_recognition)
    rgb_frame = frame[:, :, ::-1]

    # Trouver les visages dans l'image actuelle
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Boucler sur chaque visage trouvé
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Vérifier si le visage correspond à un visage connu
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # S'il y a au moins une correspondance
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Dessiner un rectangle autour du visage et afficher le nom
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Afficher l'image avec les résultats
    cv2.imshow('Video', frame)

    # Quitter la boucle si la touche 'q' est pressée
    if cv2.waitKey(1) == ord('q'):
        break

# Libérer les ressources
video_capture.release()
cv2.destroyAllWindows()