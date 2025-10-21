"""
Backend FastAPI avec 3D Gaussian Splatting (Alternative ultra-rapide)
Utilise le repo officiel: https://github.com/graphdeco-inria/gaussian-splatting
"""
import os
import uuid
import asyncio
import subprocess
import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2

# Configuration
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
JOBS_DIR = Path("jobs")
GAUSSIAN_SPLATTING_PATH = Path("/workspace/gaussian-splatting")  # À adapter

for dir_path in [UPLOAD_DIR, OUTPUT_DIR, JOBS_DIR]:
    dir_path.mkdir(exist_ok=True)

jobs_status = {}

app = FastAPI(title="3D Gaussian Splatting API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_frames_from_video(video_path: Path, output_dir: Path, fps: int = 2) -> int:
    """Extrait des frames d'une vidéo à un FPS donné"""
    output_dir.mkdir(exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_path = output_dir / f"{saved_count:04d}.jpg"
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"✅ {saved_count} frames extraites")
    return saved_count


async def process_gaussian_splatting(job_id: str, video_path: Path):
    """
    Traitement avec 3D Gaussian Splatting
    Beaucoup plus rapide que NeRF: ~1-2 minutes sur RTX 4090
    """
    try:
        jobs_status[job_id]["status"] = "processing"
        jobs_status[job_id]["message"] = "Extraction des frames..."
        jobs_status[job_id]["progress"] = 10
        
        job_dir = JOBS_DIR / job_id
        input_dir = job_dir / "input"
        output_dir = job_dir / "output"
        
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Étape 1: Extraire frames (augmenté à 3 FPS pour vidéos courtes)
        num_frames = extract_frames_from_video(video_path, input_dir, fps=3)
        
        if num_frames < 10:
            raise Exception(f"Pas assez de frames: {num_frames}. Vidéo trop courte (minimum 5 secondes)")
        
        jobs_status[job_id]["progress"] = 20
        jobs_status[job_id]["message"] = "COLMAP: Structure from Motion..."
        
        # Étape 2: COLMAP pour les poses de caméra
        colmap_script = GAUSSIAN_SPLATTING_PATH / "convert.py"
        
        colmap_cmd = [
            "python", str(colmap_script),
            "-s", str(job_dir),
            "--skip_matching"  # Plus rapide, utilise l'ordre séquentiel
        ]
        
        print(f"🔄 COLMAP: {' '.join(colmap_cmd)}")
        
        process = await asyncio.create_subprocess_exec(
            *colmap_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(GAUSSIAN_SPLATTING_PATH)
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"COLMAP failed: {stderr.decode()}")
        
        jobs_status[job_id]["progress"] = 40
        jobs_status[job_id]["message"] = "Entraînement Gaussian Splatting..."
        
        # Étape 3: Entraînement Gaussian Splatting
        train_cmd = [
            "python", "train.py",
            "-s", str(job_dir),
            "-m", str(output_dir),
            "--iterations", "7000",  # ~1-2 min sur RTX 4090
            "--test_iterations", "7000",
            "--save_iterations", "7000",
            "--quiet"
        ]
        
        print(f"🔄 Training: {' '.join(train_cmd)}")
        
        process = await asyncio.create_subprocess_exec(
            *train_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(GAUSSIAN_SPLATTING_PATH)
        )
        
        # Progression pendant l'entraînement
        progress = 40
        while process.returncode is None:
            await asyncio.sleep(3)
            progress = min(progress + 5, 85)
            jobs_status[job_id]["progress"] = progress
            
            try:
                await asyncio.wait_for(process.wait(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"Training failed: {stderr.decode()}")
        
        jobs_status[job_id]["progress"] = 90
        jobs_status[job_id]["message"] = "Export du modèle PLY..."
        
        # Étape 4: Le modèle PLY est déjà généré par Gaussian Splatting
        ply_source = output_dir / "point_cloud" / "iteration_7000" / "point_cloud.ply"
        
        if not ply_source.exists():
            raise Exception("Fichier PLY non trouvé après l'entraînement")
        
        # Copier vers le dossier de sortie
        ply_final = OUTPUT_DIR / f"{job_id}.ply"
        shutil.copy(str(ply_source), str(ply_final))
        
        # Succès
        jobs_status[job_id]["status"] = "completed"
        jobs_status[job_id]["progress"] = 100
        jobs_status[job_id]["message"] = "Modèle 3D généré avec succès"
        jobs_status[job_id]["download_url"] = f"/download/{job_id}.ply"
        
        print(f"✅ Job {job_id} terminé")
        
        # Nettoyer
        shutil.rmtree(job_dir, ignore_errors=True)
        if video_path.exists():
            video_path.unlink()
        
    except Exception as e:
        print(f"❌ Erreur job {job_id}: {str(e)}")
        jobs_status[job_id]["status"] = "failed"
        jobs_status[job_id]["message"] = "Échec de la génération"
        jobs_status[job_id]["error"] = str(e)


@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "3D Gaussian Splatting API",
        "version": "1.0.0",
        "gpu": "RTX 4090",
        "technology": "3D Gaussian Splatting"
    }


@app.post("/generate-3d")
async def generate_3d(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    try:
        if not file.content_type.startswith("video/"):
            raise HTTPException(400, "Le fichier doit être une vidéo")
        
        job_id = str(uuid.uuid4())
        video_path = UPLOAD_DIR / f"{job_id}.mp4"
        
        with open(video_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        jobs_status[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "message": "En attente...",
            "progress": 0,
            "created_at": datetime.now().isoformat(),
            "download_url": None,
            "error": None
        }
        
        background_tasks.add_task(process_gaussian_splatting, job_id, video_path)
        
        return {
            "success": True,
            "job_id": job_id,
            "message": "Génération démarrée",
            "estimated_time": "1-2 minutes"
        }
        
    except Exception as e:
        raise HTTPException(500, f"Erreur: {str(e)}")


@app.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in jobs_status:
        raise HTTPException(404, "Job non trouvé")
    return jobs_status[job_id]


@app.get("/download/{filename}")
async def download_model(filename: str):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(404, "Fichier non trouvé")
    
    return FileResponse(
        path=str(file_path),
        media_type="application/octet-stream",
        filename=filename
    )


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 3D Gaussian Splatting API")
    print("📍 http://0.0.0.0:8000")
    print("⚡ Ultra-rapide: 1-2 min sur RTX 4090")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
