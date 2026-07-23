import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import timm
from PIL import Image
from torchvision import transforms
import io
import sqlite3
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# ---------------------------------------------------------
# FEEDBACK DATABASE (local SQLite for now; swap to Supabase Postgres later)
# ---------------------------------------------------------
DATA_DIR = "data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
DB_PATH = os.path.join(DATA_DIR, "feedback.db")
os.makedirs(IMAGES_DIR, exist_ok=True)


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            image_path TEXT NOT NULL,
            model1_diag TEXT, model1_cataract REAL, model1_normal REAL, model1_noteye REAL,
            model2_diag TEXT, model2_cataract REAL, model2_normal REAL, model2_noteye REAL,
            model3_diag TEXT, model3_cataract REAL, model3_normal REAL, model3_noteye REAL,
            ensemble_diag TEXT,
            doctor_label TEXT,
            doctor_labeled_at TEXT
        )
    """)
    conn.commit()
    conn.close()


init_db()

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
def require_env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name} (set it in .env)")
    return value


CHECKPOINT_FILES = [
    require_env("CHECKPOINT_EXP1"),
    require_env("CHECKPOINT_EXP2"),
    require_env("CHECKPOINT_EXP3"),
]
NUM_CLASSES = 3  # 0: Cataract, 1: Normal, 2: Not Eye
CLASS_NAMES = {0: "Cataract", 1: "Normal", 2: "Not Eye"}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"-> Starting Backend on Device: {device.type.upper()}")

# ---------------------------------------------------------
# FASTAPI SETUP
# ---------------------------------------------------------
app = FastAPI(title="Cataract AI Diagnostician")

# Allow the deployed frontend (Vercel domain) to call this API cross-origin.
# Set ALLOWED_ORIGINS in .env, comma-separated, e.g.:
# ALLOWED_ORIGINS=https://your-app.vercel.app,http://localhost:5000
from fastapi.middleware.cors import CORSMiddleware

allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:5000,http://127.0.0.1:5000").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

# ---------------------------------------------------------
# MODEL ARCHITECTURE (Must match perfectly)
# ---------------------------------------------------------
class EyeDiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('convnext_tiny', pretrained=False, num_classes=0)
        num_features = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# Load ensemble globally
models = []
print("-> Loading Ensemble Models into memory...")
for ckpt in CHECKPOINT_FILES:
    if os.path.exists(ckpt):
        m = EyeDiseaseModel(NUM_CLASSES)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        m.to(device)
        m.eval()
        models.append(m)
        print(f"Loaded: {os.path.basename(ckpt)}")
    else:
        print(f"ERROR: Missing Checkpoint {ckpt}!")

if len(models) < 3:
    print("WARNING: Not all 3 models were loaded. Ensemble might be incomplete.")

# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------
def condition_image(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(denoised)
    colorblind_rgb = cv2.merge((cl, cl, cl))
    return colorblind_rgb

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------
# PREDICTION ENDPOINT
# ---------------------------------------------------------
from fastapi.responses import StreamingResponse
import json

@app.post("/predict")
async def predict_cataract(file: UploadFile = File(...)):
    if not models:
        return {"error": "No models loaded in the backend."}

    image_bytes = await file.read()

    async def generate():
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            tensor = val_transform(pil_img).unsqueeze(0).to(device)

            def format_probs(p):
                return {
                    "Cataract": round(float(p[0]) * 100, 2),
                    "Normal": round(float(p[1]) * 100, 2),
                    "Not Eye": round(float(p[2]) * 100, 2),
                    "Diagnosis": CLASS_NAMES[int(np.argmax(p))]
                }

            probs = []
            with torch.no_grad():
                for i, m in enumerate(models, start=1):
                    out = m(tensor)
                    p = F.softmax(out, dim=1).cpu().numpy()[0]
                    probs.append(p)
                    yield json.dumps({"model": i, "result": format_probs(p)}) + "\n"

            # Soft-vote ensemble across the loaded models
            ensemble_probs = np.mean(probs, axis=0)
            ensemble_diag = CLASS_NAMES[int(np.argmax(ensemble_probs))]

            # Persist image + predictions for doctor feedback / future fine-tuning
            image_filename = f"{uuid.uuid4().hex}.jpg"
            image_path = os.path.join(IMAGES_DIR, image_filename)
            pil_img.save(image_path, format="JPEG")

            m_results = [format_probs(p) for p in probs]
            conn = get_db()
            cur = conn.execute(
                """INSERT INTO predictions (
                    created_at, image_path,
                    model1_diag, model1_cataract, model1_normal, model1_noteye,
                    model2_diag, model2_cataract, model2_normal, model2_noteye,
                    model3_diag, model3_cataract, model3_normal, model3_noteye,
                    ensemble_diag
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(), image_path,
                    m_results[0]["Diagnosis"], m_results[0]["Cataract"], m_results[0]["Normal"], m_results[0]["Not Eye"],
                    m_results[1]["Diagnosis"], m_results[1]["Cataract"], m_results[1]["Normal"], m_results[1]["Not Eye"],
                    m_results[2]["Diagnosis"], m_results[2]["Cataract"], m_results[2]["Normal"], m_results[2]["Not Eye"],
                    ensemble_diag,
                )
            )
            conn.commit()
            prediction_id = cur.lastrowid
            conn.close()

            yield json.dumps({"done": True, "id": prediction_id, "ensemble_diagnosis": ensemble_diag}) + "\n"

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield json.dumps({"error": str(e)}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


class FeedbackRequest(BaseModel):
    id: int
    doctor_label: str


@app.post("/feedback")
async def submit_feedback(body: FeedbackRequest):
    if body.doctor_label not in ("Cataract", "Normal"):
        raise HTTPException(status_code=400, detail="doctor_label must be 'Cataract' or 'Normal'")

    conn = get_db()
    cur = conn.execute("SELECT id FROM predictions WHERE id = ?", (body.id,))
    if cur.fetchone() is None:
        conn.close()
        raise HTTPException(status_code=404, detail="Prediction not found")

    conn.execute(
        "UPDATE predictions SET doctor_label = ?, doctor_labeled_at = ? WHERE id = ?",
        (body.doctor_label, datetime.now(timezone.utc).isoformat(), body.id)
    )
    conn.commit()
    conn.close()
    return {"success": True}


@app.get("/stats")
async def get_stats():
    conn = get_db()
    rows = conn.execute(
        "SELECT ensemble_diag, model1_diag, model2_diag, model3_diag, doctor_label "
        "FROM predictions WHERE doctor_label IS NOT NULL"
    ).fetchall()
    conn.close()

    total = len(rows)
    if total == 0:
        return {
            "total_labeled": 0,
            "ensemble_accuracy": None,
            "per_model_accuracy": {"model1": None, "model2": None, "model3": None}
        }

    def accuracy(key):
        correct = sum(1 for r in rows if r[key] == r["doctor_label"])
        return round(correct / total * 100, 2)

    return {
        "total_labeled": total,
        "ensemble_accuracy": accuracy("ensemble_diag"),
        "per_model_accuracy": {
            "model1": accuracy("model1_diag"),
            "model2": accuracy("model2_diag"),
            "model3": accuracy("model3_diag"),
        }
    }

if __name__ == "__main__":
    import uvicorn
    # Optional programmatic start
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)
