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
python face_recognition/main.py
```

- Video importer UI:

```bash
python face_recognition/UI_video_processor.py
```

- CLI image import (select a face and name it):

```bash
python face_recognition/import_image.py
```

## Notes

- Encodings are stored in `face_recognition/encodings/`.
- The webcam and audio alarm require a working camera and audio output.

## Tests

```bash
pytest -q
```
