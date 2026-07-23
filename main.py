import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
import timm
from PIL import Image
from torchvision import transforms
import io
import uuid
import requests
import psycopg2
import psycopg2.extras
from datetime import datetime, timezone

from dotenv import load_dotenv

# Load variables from .env
load_dotenv()


def require_env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name} (set it in .env)")
    return value


# ---------------------------------------------------------
# FEEDBACK DATABASE (Supabase Postgres, via the connection pooler)
# ---------------------------------------------------------
SUPABASE_DB_URL = require_env("SUPABASE_DB_URL")

# ---------------------------------------------------------
# SUPABASE STORAGE (private bucket for eye images)
# ---------------------------------------------------------
SUPABASE_URL = require_env("SUPABASE_URL").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = require_env("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "cataract-images")


def _storage_auth_headers():
    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
    }


def upload_image_to_storage(image_bytes):
    object_path = f"{uuid.uuid4().hex}.jpg"
    resp = requests.post(
        f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{object_path}",
        headers={**_storage_auth_headers(), "Content-Type": "image/jpeg"},
        data=image_bytes,
        timeout=30,
    )
    resp.raise_for_status()
    return object_path


def download_image_from_storage(object_path):
    resp = requests.get(
        f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{object_path}",
        headers=_storage_auth_headers(),
        timeout=30,
    )
    resp.raise_for_status()
    return resp.content


def get_db():
    return psycopg2.connect(SUPABASE_DB_URL)


def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL,
            image_path TEXT,
            model1_diag TEXT, model1_cataract REAL, model1_normal REAL, model1_noteye REAL,
            model2_diag TEXT, model2_cataract REAL, model2_normal REAL, model2_noteye REAL,
            model3_diag TEXT, model3_cataract REAL, model3_normal REAL, model3_noteye REAL,
            ensemble_diag TEXT,
            doctor_label TEXT,
            doctor_labeled_at TIMESTAMPTZ
        )
    """)
    # Migrate from the old bytea-in-Postgres approach: add image_path if this
    # table pre-dates it, backfill any existing rows into the storage bucket,
    # then drop the old column.
    cur.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS image_path TEXT")
    cur.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'predictions' AND column_name = 'image_data'
    """)
    if cur.fetchone() is not None:
        cur.execute("SELECT id, image_data FROM predictions WHERE image_path IS NULL AND image_data IS NOT NULL")
        for row_id, image_data in cur.fetchall():
            object_path = upload_image_to_storage(bytes(image_data))
            cur.execute("UPDATE predictions SET image_path = %s WHERE id = %s", (object_path, row_id))
        cur.execute("ALTER TABLE predictions DROP COLUMN image_data")

    # share_token: a random, unguessable ID for the doctor-review link, distinct
    # from the sequential `id` (which must never appear in a shareable URL, since
    # sequential IDs could be enumerated to browse other patients' images).
    cur.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS share_token TEXT")
    cur.execute("SELECT id FROM predictions WHERE share_token IS NULL")
    for (row_id,) in cur.fetchall():
        cur.execute("UPDATE predictions SET share_token = %s WHERE id = %s", (uuid.uuid4().hex, row_id))
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS predictions_share_token_idx ON predictions (share_token)")

    conn.commit()
    cur.close()
    conn.close()


init_db()

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
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
        m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
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
            img_buffer = io.BytesIO()
            pil_img.save(img_buffer, format="JPEG")
            image_data = img_buffer.getvalue()
            object_path = upload_image_to_storage(image_data)
            share_token = uuid.uuid4().hex

            m_results = [format_probs(p) for p in probs]
            conn = get_db()
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO predictions (
                    created_at, image_path, share_token,
                    model1_diag, model1_cataract, model1_normal, model1_noteye,
                    model2_diag, model2_cataract, model2_normal, model2_noteye,
                    model3_diag, model3_cataract, model3_normal, model3_noteye,
                    ensemble_diag
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id""",
                (
                    datetime.now(timezone.utc), object_path, share_token,
                    m_results[0]["Diagnosis"], m_results[0]["Cataract"], m_results[0]["Normal"], m_results[0]["Not Eye"],
                    m_results[1]["Diagnosis"], m_results[1]["Cataract"], m_results[1]["Normal"], m_results[1]["Not Eye"],
                    m_results[2]["Diagnosis"], m_results[2]["Cataract"], m_results[2]["Normal"], m_results[2]["Not Eye"],
                    ensemble_diag,
                )
            )
            prediction_id = cur.fetchone()[0]
            conn.commit()
            cur.close()
            conn.close()

            yield json.dumps({
                "done": True,
                "id": prediction_id,
                "ensemble_diagnosis": ensemble_diag,
                "share_token": share_token,
            }) + "\n"

        except Exception:
            import traceback
            traceback.print_exc()
            yield json.dumps({"error": "An internal error occurred while processing the image."}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.get("/stats")
async def get_stats():
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        "SELECT ensemble_diag, model1_diag, model2_diag, model3_diag, doctor_label "
        "FROM predictions WHERE doctor_label IS NOT NULL"
    )
    rows = cur.fetchall()
    cur.close()
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


@app.get("/gallery", response_class=HTMLResponse)
async def gallery():
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        "SELECT id, created_at, model1_diag, model2_diag, model3_diag, "
        "ensemble_diag, doctor_label FROM predictions ORDER BY id DESC"
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    cards = []
    for r in rows:
        label = r["doctor_label"] or "Not labeled yet"
        label_class = "labeled" if r["doctor_label"] else "unlabeled"
        date = r["created_at"].strftime("%Y-%m-%d %H:%M:%S")
        cards.append(f"""
        <div class="card">
            <img src="/images/{r['id']}" loading="lazy" alt="Prediction {r['id']}">
            <div class="info">
                <div class="row"><span>ID</span><strong>{r['id']}</strong></div>
                <div class="row"><span>Model 1</span><strong>{r['model1_diag']}</strong></div>
                <div class="row"><span>Model 2</span><strong>{r['model2_diag']}</strong></div>
                <div class="row"><span>Model 3</span><strong>{r['model3_diag']}</strong></div>
                <div class="row"><span>Ensemble</span><strong>{r['ensemble_diag']}</strong></div>
                <div class="row label {label_class}"><span>Doctor</span><strong>{label}</strong></div>
                <div class="row"><span>Date</span><strong>{date}</strong></div>
            </div>
        </div>
        """)

    html = f"""<!DOCTYPE html>
<html><head><title>Prediction Gallery</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
    body {{ font-family: sans-serif; background:#0f111a; color:#f8fafc; padding:2rem; margin:0; }}
    h1 {{ margin-bottom: 1.5rem; }}
    .grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(240px,1fr)); gap:1.5rem; }}
    .card {{ background: rgba(25,28,41,0.8); border:1px solid rgba(255,255,255,0.08); border-radius:16px; overflow:hidden; }}
    .card img {{ width:100%; height:200px; object-fit:cover; display:block; }}
    .info {{ padding: 0.75rem 1rem; font-size: 0.85rem; }}
    .row {{ display:flex; justify-content:space-between; padding: 2px 0; }}
    .row span {{ color:#94a3b8; }}
    .label.labeled strong {{ color:#10b981; }}
    .label.unlabeled strong {{ color:#f59e0b; }}
</style></head>
<body>
    <h1>Prediction Gallery ({len(rows)} total)</h1>
    <div class="grid">{"".join(cards)}</div>
</body></html>"""
    return HTMLResponse(html)


class TokenFeedbackRequest(BaseModel):
    doctor_label: str


@app.get("/review/{token}", response_class=HTMLResponse)
async def review(token: str):
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        "SELECT created_at, model1_diag, model1_cataract, model2_diag, model2_cataract, "
        "model3_diag, model3_cataract, ensemble_diag, doctor_label "
        "FROM predictions WHERE share_token = %s",
        (token,)
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Not found")

    date = row["created_at"].strftime("%Y-%m-%d %H:%M:%S")
    if row["doctor_label"]:
        feedback_html = f'<div class="current-label">Already labeled: <strong>{row["doctor_label"]}</strong></div>'
    else:
        feedback_html = """
        <div class="feedback-buttons">
            <button id="btn-cataract" onclick="submitLabel('Cataract')">Cataract</button>
            <button id="btn-normal" onclick="submitLabel('Normal')">Normal</button>
        </div>
        <p id="status"></p>
        """

    html = f"""<!DOCTYPE html>
<html><head><title>Patient Eye Review</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
    body {{ font-family: sans-serif; background:#0f111a; color:#f8fafc; padding:2rem; margin:0; display:flex; justify-content:center; }}
    .panel {{ max-width: 480px; width:100%; }}
    img {{ width:100%; border-radius:16px; margin-bottom:1.5rem; }}
    h1 {{ font-size:1.4rem; margin-bottom:1rem; }}
    .row {{ display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid rgba(255,255,255,0.08); }}
    .row span {{ color:#94a3b8; }}
    .feedback-buttons {{ display:flex; gap:0.75rem; margin-top:1.5rem; }}
    button {{ flex:1; padding:12px; border:none; border-radius:10px; font-size:1rem; font-weight:600; cursor:pointer; }}
    #btn-cataract {{ background:#ef4444; color:white; }}
    #btn-normal {{ background:#10b981; color:white; }}
    button:disabled {{ opacity:0.5; cursor:not-allowed; }}
    .current-label {{ margin-top:1.5rem; padding:1rem; background:rgba(16,185,129,0.15); border-radius:10px; text-align:center; font-weight:600; }}
    #status {{ margin-top:1rem; text-align:center; color:#10b981; font-weight:600; }}
</style></head>
<body>
    <div class="panel">
        <h1>Patient Eye Image Review</h1>
        <img src="/review/{token}/image" alt="Eye image">
        <div class="row"><span>Model 1</span><strong>{row['model1_diag']} ({row['model1_cataract']}% cataract)</strong></div>
        <div class="row"><span>Model 2</span><strong>{row['model2_diag']} ({row['model2_cataract']}% cataract)</strong></div>
        <div class="row"><span>Model 3</span><strong>{row['model3_diag']} ({row['model3_cataract']}% cataract)</strong></div>
        <div class="row"><span>Date</span><strong>{date}</strong></div>
        {feedback_html}
    </div>
    <script>
        async function submitLabel(label) {{
            document.getElementById('btn-cataract').disabled = true;
            document.getElementById('btn-normal').disabled = true;
            try {{
                const resp = await fetch('/review/{token}/feedback', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{doctor_label: label}})
                }});
                if (!resp.ok) throw new Error('failed');
                document.getElementById('status').textContent = 'Saved: ' + label;
            }} catch (e) {{
                document.getElementById('status').textContent = 'Failed to save, please try again.';
                document.getElementById('btn-cataract').disabled = false;
                document.getElementById('btn-normal').disabled = false;
            }}
        }}
    </script>
</body></html>"""
    return HTMLResponse(html)


@app.get("/review/{token}/image")
async def review_image(token: str):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT image_path FROM predictions WHERE share_token = %s", (token,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row is None or row[0] is None:
        raise HTTPException(status_code=404, detail="Image not found")
    image_data = download_image_from_storage(row[0])
    return Response(content=image_data, media_type="image/jpeg")


@app.post("/review/{token}/feedback")
async def review_feedback(token: str, body: TokenFeedbackRequest):
    if body.doctor_label not in ("Cataract", "Normal"):
        raise HTTPException(status_code=400, detail="doctor_label must be 'Cataract' or 'Normal'")

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id FROM predictions WHERE share_token = %s", (token,))
    row = cur.fetchone()
    if row is None:
        cur.close()
        conn.close()
        raise HTTPException(status_code=404, detail="Prediction not found")

    cur.execute(
        "UPDATE predictions SET doctor_label = %s, doctor_labeled_at = %s WHERE id = %s",
        (body.doctor_label, datetime.now(timezone.utc), row[0])
    )
    conn.commit()
    cur.close()
    conn.close()
    return {"success": True}


@app.get("/images/{prediction_id}")
async def get_image(prediction_id: int):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT image_path FROM predictions WHERE id = %s", (prediction_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row is None or row[0] is None:
        raise HTTPException(status_code=404, detail="Image not found")
    image_data = download_image_from_storage(row[0])
    return Response(content=image_data, media_type="image/jpeg")

# Serve the frontend at the root, matching how Vercel serves the `static/`
# folder as its own root directory (same file paths work in both places).
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # Optional programmatic start
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)
