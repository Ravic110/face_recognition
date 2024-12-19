import use_webCam, import_image, import_video

import verify_if_exist

def main():
    print("1. Capture depuis la webcam")
    print("2. Importer une image")
    print("3. Importer une vidéo")
    print("4. Vérifier si un visage existe")

    choice = input("Choisissez une option (1, 2, 3 ou 4) : ")

    if choice == '1':
        name = input("Entrez le nom de la personne : ")
        encoder = use_webCam.SecureFaceEncoder()
        encoder.capture_face(name)
    elif choice == '2':
        name = input("Entrez le nom de la personne : ")
        import_image.import_face_from_image(name)
    elif choice == '3':
        import_video.import_video()
    elif choice == '4':
        verify_if_exist.import_and_check_face()
    else:
        print("Option invalide.")

if __name__ == "__main__":
    main()
