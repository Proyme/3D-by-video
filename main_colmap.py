"""
Backend FastAPI avec COLMAP uniquement (comme en local)
Plus simple et plus stable que Gaussian Splatting
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

for dir_path in [UPLOAD_DIR, OUTPUT_DIR, JOBS_DIR]:
    dir_path.mkdir(exist_ok=True)

jobs_status = {}

app = FastAPI(title="3D COLMAP API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_frames_from_video(video_path: Path, output_dir: Path, fps: int = 3) -> int:
    """Extrait des frames d'une vidéo"""
    output_dir.mkdir(exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(video_fps / fps))
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_path = output_dir / f"frame_{saved_count:04d}.jpg"
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"✅ {saved_count} frames extraites")
    return saved_count


async def process_colmap(job_id: str, video_path: Path):
    """
    Traitement avec COLMAP uniquement
    Simple et stable
    """
    try:
        jobs_status[job_id]["status"] = "processing"
        jobs_status[job_id]["message"] = "Extraction des frames..."
        jobs_status[job_id]["progress"] = 10
        
        job_dir = JOBS_DIR / job_id
        images_dir = job_dir / "images"
        sparse_dir = job_dir / "sparse"
        database_path = job_dir / "database.db"
        
        images_dir.mkdir(parents=True, exist_ok=True)
        sparse_dir.mkdir(parents=True, exist_ok=True)
        
        # Étape 1: Extraire frames
        num_frames = extract_frames_from_video(video_path, images_dir, fps=3)
        
        if num_frames < 10:
            raise Exception(f"Pas assez de frames: {num_frames}. Vidéo trop courte (minimum 5 secondes)")
        
        jobs_status[job_id]["progress"] = 20
        jobs_status[job_id]["message"] = "COLMAP: Feature extraction..."
        
        # Étape 2: COLMAP Feature Extraction
        print("🔄 COLMAP: Feature extraction")
        feature_cmd = [
            "colmap", "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(images_dir),
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", "SIMPLE_RADIAL"
        ]
        
        result = subprocess.run(feature_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Feature extraction failed: {result.stderr}")
        
        jobs_status[job_id]["progress"] = 40
        jobs_status[job_id]["message"] = "COLMAP: Feature matching..."
        
        # Étape 3: COLMAP Feature Matching
        print("🔄 COLMAP: Feature matching")
        matching_cmd = [
            "colmap", "exhaustive_matcher",
            "--database_path", str(database_path)
        ]
        
        result = subprocess.run(matching_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Feature matching failed: {result.stderr}")
        
        jobs_status[job_id]["progress"] = 60
        jobs_status[job_id]["message"] = "COLMAP: Reconstruction 3D..."
        
        # Étape 4: COLMAP Mapper (Reconstruction)
        print("🔄 COLMAP: Mapper")
        mapper_cmd = [
            "colmap", "mapper",
            "--database_path", str(database_path),
            "--image_path", str(images_dir),
            "--output_path", str(sparse_dir)
        ]
        
        result = subprocess.run(mapper_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Mapper failed: {result.stderr}")
        
        jobs_status[job_id]["progress"] = 80
        jobs_status[job_id]["message"] = "Export du modèle 3D..."
        
        # Étape 5: Export PLY
        print("🔄 Export PLY")
        model_dir = sparse_dir / "0"  # COLMAP crée un dossier "0" par défaut
        
        if not model_dir.exists():
            raise Exception("Reconstruction failed: no model generated")
        
        output_ply = OUTPUT_DIR / f"{job_id}.ply"
        
        export_cmd = [
            "colmap", "model_converter",
            "--input_path", str(model_dir),
            "--output_path", str(output_ply),
            "--output_type", "PLY"
        ]
        
        result = subprocess.run(export_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Export failed: {result.stderr}")
        
        # Nettoyage
        video_path.unlink(missing_ok=True)
        shutil.rmtree(job_dir, ignore_errors=True)
        
        jobs_status[job_id]["status"] = "completed"
        jobs_status[job_id]["message"] = "Modèle 3D généré !"
        jobs_status[job_id]["progress"] = 100
        jobs_status[job_id]["download_url"] = f"/download/{job_id}.ply"
        
        print(f"✅ Job {job_id} terminé")
        
    except Exception as e:
        print(f"❌ Erreur job {job_id}: {str(e)}")
        jobs_status[job_id]["status"] = "failed"
        jobs_status[job_id]["message"] = f"Erreur: {str(e)}"
        jobs_status[job_id]["error"] = str(e)


@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "online",
        "service": "3D COLMAP API",
        "version": "1.0.0",
        "technology": "COLMAP Structure from Motion"
    }


@app.post("/generate-3d")
async def generate_3d(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...)
):
    """Upload vidéo et démarre la génération 3D"""
    
    # Générer un ID unique
    job_id = str(uuid.uuid4())
    
    # Sauvegarder la vidéo
    video_path = UPLOAD_DIR / f"{job_id}.mp4"
    
    with open(video_path, "wb") as f:
        content = await video.read()
        f.write(content)
    
    # Initialiser le statut
    jobs_status[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "message": "En attente...",
        "progress": 0,
        "created_at": datetime.now().isoformat(),
        "download_url": None,
        "error": None
    }
    
    # Lancer le traitement en arrière-plan
    background_tasks.add_task(process_colmap, job_id, video_path)
    
    return {
        "success": True,
        "job_id": job_id,
        "message": "Génération démarrée",
        "estimated_time": "2-5 minutes"
    }


@app.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    """Récupère le statut d'un job"""
    if job_id not in jobs_status:
        raise HTTPException(status_code=404, detail="Job non trouvé")
    
    return jobs_status[job_id]


@app.get("/download/{filename}")
async def download_model(filename: str):
    """Télécharge un modèle 3D"""
    file_path = OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Fichier non trouvé")
    
    return FileResponse(
        path=file_path,
        media_type="application/octet-stream",
        filename=filename
    )


@app.get("/jobs")
async def list_jobs():
    """Liste tous les jobs"""
    return {"jobs": list(jobs_status.values())}


@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Supprime un job et ses fichiers"""
    if job_id in jobs_status:
        # Supprimer les fichiers
        output_file = OUTPUT_DIR / f"{job_id}.ply"
        output_file.unlink(missing_ok=True)
        
        job_dir = JOBS_DIR / job_id
        shutil.rmtree(job_dir, ignore_errors=True)
        
        del jobs_status[job_id]
        
        return {"success": True, "message": "Job supprimé"}
    
    raise HTTPException(status_code=404, detail="Job non trouvé")


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 3D COLMAP API")
    print("📍 http://0.0.0.0:8000")
    print("⚡ Reconstruction 3D avec COLMAP")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
