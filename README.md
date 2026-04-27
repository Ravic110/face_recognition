# Face Recognition Project

Local face recognition tools with a Tkinter UI, webcam mode, and video importer.

## Setup

- Create and activate a virtual environment
- Install dependencies:

```bash
pip install -r requirements.txt
```

- Optional: install the package in editable mode for easier imports and CLI usage:

```bash
pip install -e .
```

## Run

- Main launcher (image UI + realtime webcam):

```bash
python main.py
```

- Package entrypoint (after `pip install -e .`):

```bash
python -m face_recognition_app
```

- Video importer UI:

```bash
python -m face_recognition_app.ui.video_importer
```

- CLI image import (select a face and name it):

```bash
python -m face_recognition_app.ui.import_image
```

## Project structure

- `main.py` — launcher local qui ajoute `src` au `PYTHONPATH` et démarre l'application.
- `src/face_recognition_app/` — package principal du projet : UI, services, stockage.
- `encodings/` — stockage des encodages de visages au format JSON.
- `events.db` — base SQLite des événements de surveillance.
- `clips/` — clips vidéo générés par le système (si l'enregistrement est activé).

## Notes

- Encodings are stored in `encodings/`.
- The webcam and audio alarm require a working camera and audio output.
- The project is still under active development and some modules are prototype-level.

## Tests

```bash
pytest -q
```
