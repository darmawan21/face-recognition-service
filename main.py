import os
import cv2
import numpy as np
import tensorflow as tf
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
from typing import List, Optional
import json

# ── Model state ──────────────────────────────────────────────────────────────
interpreter = None
input_details = None
output_details = None

MODEL_PATH = os.getenv("MODEL_PATH", "mobilefacenet.tflite")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load TFLite model once on startup, release on shutdown."""
    global interpreter, input_details, output_details
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file '{MODEL_PATH}' not found.")
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"[startup] TFLite model loaded: {MODEL_PATH}")
    yield
    print("[shutdown] Cleaning up resources.")


app = FastAPI(
    title="Face Verification Microservice",
    description="Extracts 192D MobileFaceNet embeddings and verifies faces via cosine similarity.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
# Adjust ALLOWED_ORIGINS in your .env or pass via environment variable.
# Default: allow all origins (suitable for internal/intranet use).
_raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS: List[str] = (
    ["*"] if _raw_origins == "*" else [o.strip() for o in _raw_origins.split(",")]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Cosine similarity threshold ──────────────────────────────────────────
VERIFY_THRESHOLD = 0.75

def _detect_and_extract(img: np.ndarray) -> list:
    """
    Detect face in image, crop, preprocess, run TFLite, return 192D embedding.
    Shared by /extract-embedding and /verify endpoints.
    """
    # 1. Detect Face using DeepFace (OpenCV)
    try:
        face_objs = DeepFace.extract_faces(
            img_path=img,
            detector_backend="opencv",
            enforce_detection=True,
            align=False
        )
    except ValueError as e:
        if "Face could not be detected" in str(e):
            raise HTTPException(status_code=400, detail="No face detected in the image.")
        raise

    if not face_objs or len(face_objs) == 0:
        raise HTTPException(status_code=400, detail="No face detected in the image.")

    area = face_objs[0]["facial_area"]
    x, y, w, h = area['x'], area['y'], area['w'], area['h']

    # 2. Shrink OpenCV's full-head crop to match consistent tight crop
    tight_w = int(w * 0.75)
    tight_h = int(h * 0.75)
    tight_x = x + int(w * 0.125)
    tight_y = y + int(h * 0.2)

    # Ensure bounds are within the image
    img_h, img_w = img.shape[:2]
    tight_x = max(0, min(tight_x, img_w - 1))
    tight_y = max(0, min(tight_y, img_h - 1))
    tight_w = max(1, min(tight_w, img_w - tight_x))
    tight_h = max(1, min(tight_h, img_h - tight_y))

    face_bgr = img[tight_y:tight_y+tight_h, tight_x:tight_x+tight_w]

    # Convert BGR to RGB
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

    # Resize to 112x112
    face_resized = cv2.resize(face_rgb, (112, 112))

    # Normalize: [0,255] uint8 → [-1, 1] float32
    face_preprocessed = (face_resized.astype(np.float32) - 127.5) / 127.5

    # Expand dims to [1, 112, 112, 3]
    input_data = np.expand_dims(face_preprocessed, axis=0)

    # 3. Run TFLite inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    embedding = output_data[0]

    # 4. L2 Normalization
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / (norm + 1e-10)

    return embedding.tolist()


def _cosine_similarity(a: list, b: list) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


@app.get("/")
def read_root():
    return {"status": "Face Verification Microservice is running", "version": "1.0.0"}


@app.get("/health")
def health_check():
    """Kubernetes / load-balancer liveness probe endpoint."""
    if interpreter is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {"status": "ok"}


@app.post("/extract-embedding")
async def extract_embedding(file: UploadFile = File(...)):
    """
    Receives an uploaded image, detects the face, and extracts a 192D embedding.
    Used by admin web enrollment.
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Could not read the image.")

        embedding_list = _detect_and_extract(img)

        if len(embedding_list) != 192:
            raise HTTPException(status_code=500, detail=f"Expected 192D embedding, got {len(embedding_list)}D.")

        return {"embedding": embedding_list}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/verify")
async def verify_face(
    file: UploadFile = File(...),
    stored_embedding: str = Form(...)
):
    """
    Receives a face image + stored enrollment embedding (JSON array).
    Extracts embedding from the image using the SAME pipeline as enrollment,
    then computes cosine similarity.
    
    Used by the backend during clock-in/clock-out to verify the mobile face
    against the enrolled face — ensuring both go through identical preprocessing.
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        # Parse stored embedding
        try:
            enrolled = json.loads(stored_embedding)
            if not isinstance(enrolled, list) or len(enrolled) != 192:
                raise ValueError("Invalid embedding dimension")
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid stored_embedding: {e}")

        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Could not read the image.")

        # Extract embedding using same pipeline as enrollment
        extracted = _detect_and_extract(img)

        # Compute cosine similarity
        score = _cosine_similarity(extracted, enrolled)
        verified = score >= VERIFY_THRESHOLD

        return {
            "verified": verified,
            "score": round(score, 4),
            "embedding": extracted,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
