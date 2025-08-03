from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import numpy as np
import cv2, io, os, torch, requests

app = FastAPI()

# Allow frontend to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Model Setup ===
MODEL_PATH = "sam_vit_h_4b8939.pth"
MODEL_URL = "https://huggingface.co/your-repo/sam_vit_h_4b8939.pth"  # Replace with your hosted model URL
MODEL_TYPE = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Auto-download weights if missing
if not os.path.exists(MODEL_PATH):
    print("Downloading SAM weights...")
    with requests.get(MODEL_URL, stream=True) as r:
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)

print("Loading SAM model...")
sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_PATH).to(device)
predictor = SamPredictor(sam)
print("Model loaded successfully")

# === API Endpoint ===
@app.post("/extract-face")
async def extract_face(image: UploadFile = File(...)):
    img = Image.open(image.file).convert("RGB")
    img_np = np.array(img)

    predictor.set_image(img_np)
    masks, scores, _ = predictor.predict(point_coords=None, point_labels=None, multimask_output=True)
    best_mask = masks[np.argmax(scores)]

    mask_uint8 = (best_mask * 255).astype(np.uint8)
    cutout = cv2.bitwise_and(img_np, img_np, mask=mask_uint8)

    # Return the cutout image
    cutout_img = Image.fromarray(cutout)
    buf = io.BytesIO()
    cutout_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
