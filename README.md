# Face Recognition Project

Local face recognition tools with a Tkinter UI, webcam mode, and video importer.

## Setup

- Create and activate a virtual environment
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

- Main launcher (image UI + realtime webcam):

```bash
python main.py
```

- Video importer UI:

```bash
PYTHONPATH=src python -m face_recognition_app.ui.video_importer
```

- CLI image import (select a face and name it):

```bash
PYTHONPATH=src python -m face_recognition_app.ui.import_image
```

## Notes

- Encodings are stored in `encodings/`.
- The webcam and audio alarm require a working camera and audio output.

## Tests

```bash
pytest -q
```
