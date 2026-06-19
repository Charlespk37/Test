"""
Ruche Dashboard — Backend API v2 (avec vision IA)
===================================================
Routes :
  GET  /                    → health check
  POST /api/pico            → données capteurs Pico WH
  GET  /api/status          → état courant (pour le frontend)
  POST /api/video           → upload vidéo → analyse IA → retourne IN/OUT
  GET  /api/video/{job_id}  → statut/résultat d'un job d'analyse
  GET  /api/video/{job_id}/annotated → télécharger la vidéo annotée
  POST /api/frame           → image base64 (comptage vision unitaire)
"""

import os
import uuid
import json
import time
import base64
import asyncio
import logging
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Any
from concurrent.futures import ProcessPoolExecutor

from fastapi import FastAPI, Request, HTTPException, Depends, Header, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("ruche.api")

# ── Config ────────────────────────────────────────────────────────────────────
RUCHE_TOKEN    = os.getenv("RUCHE_TOKEN", "")
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")
DATA_FILE      = Path(os.getenv("DATA_FILE", "ruche_data.json"))
STALE_SECONDS  = int(os.getenv("STALE_SECONDS", "300"))
MAX_VIDEO_MB   = int(os.getenv("MAX_VIDEO_MB", "500"))
UPLOAD_DIR     = Path(os.getenv("UPLOAD_DIR", "/tmp/ruche_uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Ruche API v2", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN] if ALLOWED_ORIGIN != "*" else ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

# Executor pour CPU-bound (analyse vidéo dans un process séparé)
_executor = ProcessPoolExecutor(max_workers=2)

# Store in-memory des jobs (en production → Redis/SQLite)
_jobs: dict[str, dict] = {}

# ── Modèles ───────────────────────────────────────────────────────────────────
class PicoPayload(BaseModel):
    device_id:        str            = Field(..., example="ruche-1-pico")
    temperature_c:    Optional[float]= Field(None, ge=-40, le=85)
    humidity_percent: Optional[float]= Field(None, ge=0,  le=100)
    battery_percent:  Optional[float]= Field(None, ge=0,  le=100)
    bee_in:           Optional[int]  = Field(None, ge=0)
    bee_out:          Optional[int]  = Field(None, ge=0)
    uptime_s:         Optional[int]  = Field(None, ge=0)
    extra:            Optional[Any]  = None

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_state() -> dict:
    try: return json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except: return {}

def save_state(state: dict):
    DATA_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

async def verify_token(x_ruche_token: Optional[str] = Header(None)):
    if RUCHE_TOKEN and x_ruche_token != RUCHE_TOKEN:
        raise HTTPException(401, "Token invalide")

# ── Worker (process séparé pour ne pas bloquer FastAPI) ───────────────────────
def _run_analysis(video_path: str, line_pos: float, annotate: bool) -> dict:
    """Exécuté dans un ProcessPoolExecutor."""
    from bee_counter import BeeCounter
    counter = BeeCounter()
    result  = counter.process_video(video_path, annotate=annotate, line_position=line_pos)
    return result.to_dict() | {"annotated_path": result.annotated_video_path}

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["meta"])
async def health():
    return {"status": "ok", "service": "ruche-api", "version": "2.0.0", "ts": now_iso()}


@app.post("/api/pico", tags=["pico"], dependencies=[Depends(verify_token)])
async def receive_pico(payload: PicoPayload):
    ts = now_iso()
    state = load_state()
    state.update({
        "device_id":        payload.device_id,
        "temperature_c":    payload.temperature_c,
        "humidity_percent": payload.humidity_percent,
        "battery_percent":  payload.battery_percent,
        "bee_in":           payload.bee_in,
        "bee_out":          payload.bee_out,
        "uptime_s":         payload.uptime_s,
        "timestamp":        ts,
    })
    if payload.extra:
        state["extra"] = payload.extra
    save_state(state)
    logger.info(f"Pico data: {payload.device_id} T={payload.temperature_c} in={payload.bee_in} out={payload.bee_out}")
    return {"status": "ok", "received_at": ts}


@app.get("/api/status", tags=["frontend"])
async def get_status():
    state = load_state()
    if not state:
        return JSONResponse({"online": False, "message": "Aucune donnée reçue."})
    ts = state.get("timestamp")
    online = False
    if ts:
        try:
            age = time.time() - datetime.fromisoformat(ts).timestamp()
            online = age < STALE_SECONDS
        except: pass
    return JSONResponse({**state, "online": online})


@app.post("/api/video", tags=["vision"])
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Vidéo MP4/AVI/MOV"),
    line_position: float = Form(0.5, ge=0.0, le=1.0,
        description="Position relative de la ligne (0=haut, 1=bas)"),
    annotate: bool = Form(True, description="Générer une vidéo annotée"),
    update_state: bool = Form(True, description="Mettre à jour /api/status avec les résultats"),
):
    """
    Upload une vidéo → lance l'analyse IA en arrière-plan.
    Retourne immédiatement un job_id pour suivre l'avancement.
    """
    size = 0
    content_type = file.content_type or ""
    if not any(ext in (file.filename or "").lower() for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]):
        if "video" not in content_type:
            raise HTTPException(422, "Fichier vidéo requis (mp4, avi, mov, mkv, webm)")

    # Sauvegarde temporaire
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(dir=UPLOAD_DIR, suffix=suffix, delete=False)
    try:
        chunk = await file.read(1024 * 1024)
        while chunk:
            size += len(chunk)
            if size > MAX_VIDEO_MB * 1024 * 1024:
                tmp.close()
                os.unlink(tmp.name)
                raise HTTPException(413, f"Vidéo trop volumineuse (max {MAX_VIDEO_MB} MB)")
            tmp.write(chunk)
            chunk = await file.read(1024 * 1024)
    finally:
        tmp.close()

    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "status":     "queued",
        "created_at": now_iso(),
        "filename":   file.filename,
        "size_mb":    round(size / 1024 / 1024, 2),
    }

    async def run_job():
        _jobs[job_id]["status"] = "processing"
        _jobs[job_id]["started_at"] = now_iso()
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                _executor, _run_analysis, tmp.name, line_position, annotate
            )
            _jobs[job_id].update({
                "status":       "done",
                "finished_at":  now_iso(),
                "result":       result,
            })
            # Mise à jour de l'état global si demandé
            if update_state:
                state = load_state()
                state["bee_in"]        = (state.get("bee_in", 0) or 0) + result["bee_in"]
                state["bee_out"]       = (state.get("bee_out", 0) or 0) + result["bee_out"]
                state["timestamp"]     = now_iso()
                state["last_video_job"] = job_id
                save_state(state)
                logger.info(f"État mis à jour depuis vidéo job {job_id}: +{result['bee_in']} in, +{result['bee_out']} out")
        except Exception as e:
            _jobs[job_id].update({"status": "error", "error": str(e)})
            logger.exception(f"Erreur job {job_id}: {e}")
        finally:
            try: os.unlink(tmp.name)
            except: pass

    background_tasks.add_task(run_job)

    return JSONResponse({
        "job_id":   job_id,
        "status":   "queued",
        "poll_url": f"/api/video/{job_id}",
        "message":  "Analyse lancée. Interroge poll_url toutes les 2 s.",
    }, status_code=202)


@app.get("/api/video/{job_id}", tags=["vision"])
async def get_job_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job introuvable")
    resp = {k: v for k, v in job.items() if k != "result" or True}
    if job.get("status") == "done" and "result" in job:
        r = job["result"]
        resp["result"] = r
        if r.get("annotated_path") and Path(r["annotated_path"]).exists():
            resp["annotated_url"] = f"/api/video/{job_id}/annotated"
    return JSONResponse(resp)


@app.get("/api/video/{job_id}/annotated", tags=["vision"])
async def download_annotated(job_id: str):
    job = _jobs.get(job_id)
    if not job or job.get("status") != "done":
        raise HTTPException(404, "Vidéo annotée non disponible")
    path = job.get("result", {}).get("annotated_path")
    if not path or not Path(path).exists():
        raise HTTPException(404, "Fichier vidéo annoté introuvable")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"ruche_annotated_{job_id}.mp4",
        headers={"Content-Disposition": f'attachment; filename="ruche_annotated_{job_id}.mp4"'}
    )


@app.post("/api/frame", tags=["vision"])
async def receive_frame(request: Request):
    """Reçoit une image base64 → analyse instantanée (sans tracking inter-frames)."""
    try: body = await request.json()
    except: raise HTTPException(400, "JSON invalide")

    img_b64 = body.get("image_b64", "")
    if not img_b64:
        raise HTTPException(422, "Champ image_b64 manquant")
    try:
        img_bytes = base64.b64decode(img_b64)
    except:
        raise HTTPException(422, "image_b64 invalide")

    import cv2, numpy as np
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(422, "Impossible de décoder l'image")

    from bee_counter import BeeCounter
    counter = BeeCounter()
    dets = counter._detect_frame(frame)

    return JSONResponse({
        "status":        "ok",
        "bee_detected":  len(dets),
        "detections":    dets[:, :5].tolist() if len(dets) else [],
        "note":          "Comptage instantané sans tracking inter-frames."
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
