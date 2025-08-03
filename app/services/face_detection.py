import cv2
import numpy as np
from io import BytesIO
from PIL import Image

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_and_crop_face(image_bytes: bytes):
    try:
        # Convert bytes to numpy image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Could not decode image."}

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) == 0:
            return {"error": "No face detected."}

        # Crop first face
        x, y, w, h = faces[0]
        face_img = img[y:y+h, x:x+w]

        # Convert to BytesIO
        pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        img_bytes = BytesIO()
        pil_img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        return img_bytes

    except Exception as e:
        return {"error": str(e)}
