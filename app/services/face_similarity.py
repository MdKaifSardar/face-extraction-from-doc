from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np
from fastapi import UploadFile

# Load ArcFace model once at startup
app_face = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app_face.prepare(ctx_id=0)

def get_face_embedding(upload_file: UploadFile):
    """Extract the face embedding from an uploaded image."""
    img = Image.open(upload_file.file).convert("RGB")
    img = np.array(img)
    faces = app_face.get(img)
    if not faces:
        return None
    return faces[0].normed_embedding  # Only first detected face

def check_human_similarity(main_photo: UploadFile, test_photo: UploadFile, threshold: float = 0.3):
    """Compare the main image with a single other image."""
    main_emb = get_face_embedding(main_photo)
    if main_emb is None:
        return {"error": "No face found in main photo"}

    test_emb = get_face_embedding(test_photo)
    if test_emb is None:
        return {"error": "No face found in test photo"}

    similarity = float(np.dot(main_emb, test_emb))
    same_person = similarity > (1 - threshold)

    return {
        "similarity": similarity,
        "same_person": same_person
    }
