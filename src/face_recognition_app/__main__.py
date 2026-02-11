import tkinter as tk
from tkinter import ttk

from face_recognition_app.ui import interface, import_image, video_importer


def launch_image_interface(root):
    window = tk.Toplevel(root)
    interface.FaceRecognitionApp(window)


def launch_realtime_recognition(root):
    window = tk.Toplevel(root)
    app = interface.FaceRecognitionApp(window)
    window.after(0, app.start_camera)


def launch_video_importer(root):
    window = video_importer.ttk.Toplevel(root)
    video_importer.VideoImporterApp(window)


def launch_import_image_cli():
    import_image.import_face_from_image()


def main():
    root = tk.Tk()
    root.title("Systeme de Reconnaissance Faciale")

    root.geometry("450x340")
    root.resizable(False, False)

    frame = ttk.Frame(root, padding=20)
    frame.pack(expand=True)

    ttk.Label(frame, text="Que souhaitez-vous faire ?", font=("Arial", 14)).pack(pady=10)

    ttk.Button(
        frame,
        text="Charger une image",
        command=lambda: launch_image_interface(root),
        width=35,
    ).pack(pady=8)

    ttk.Button(
        frame,
        text="Reconnaissance en temps reel (Webcam)",
        command=lambda: launch_realtime_recognition(root),
        width=35,
    ).pack(pady=8)

    ttk.Button(
        frame,
        text="Importer une video",
        command=lambda: launch_video_importer(root),
        width=35,
    ).pack(pady=8)

    ttk.Button(
        frame,
        text="Importer une image (CLI)",
        command=launch_import_image_cli,
        width=35,
    ).pack(pady=8)

    ttk.Button(
        frame,
        text="Quitter",
        command=root.destroy,
        width=35,
    ).pack(pady=8)

    root.mainloop()


if __name__ == "__main__":
    main()
