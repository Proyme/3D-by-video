"""
Backend FastAPI pour génération 3D avec Nerfstudio sur RunPod RTX 4090
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
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np

# Configuration
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
JOBS_DIR = Path("jobs")
TEMP_DIR = Path("temp")

for dir_path in [UPLOAD_DIR, OUTPUT_DIR, JOBS_DIR, TEMP_DIR]:
    dir_path.mkdir(exist_ok=True)

# État des jobs en mémoire (en production, utiliser Redis)
jobs_status = {}

app = FastAPI(title="3D Generation API", version="1.0.0")

# CORS pour React Native
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class JobStatus(BaseModel):
    job_id: str
    status: str  # queued, processing, completed, failed
    message: str
    progress: int
    download_url: Optional[str] = None
    error: Optional[str] = None


def extract_frames_from_video(video_path: Path, output_dir: Path, target_frames: int = 50) -> int:
    """
    Extrait des frames d'une vidéo pour Nerfstudio
    
    Args:
        video_path: Chemin vers la vidéo
        output_dir: Dossier de sortie pour les frames
        target_frames: Nombre de frames à extraire
    
    Returns:
        Nombre de frames extraites
    """
    output_dir.mkdir(exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculer l'intervalle pour obtenir target_frames uniformément réparties
    interval = max(1, total_frames // target_frames)
    
    frame_count = 0
    extracted_count = 0
    
    print(f"📹 Vidéo: {total_frames} frames à {fps} FPS")
    print(f"🎯 Extraction: 1 frame tous les {interval} frames")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % interval == 0:
            # Sauvegarder la frame
            frame_path = output_dir / f"frame_{extracted_count:04d}.jpg"
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            extracted_count += 1
            
            if extracted_count >= target_frames:
                break
        
        frame_count += 1
    
    cap.release()
    print(f"✅ {extracted_count} frames extraites")
    
    return extracted_count


async def process_3d_generation(job_id: str, video_path: Path):
    """
    Traitement asynchrone de la génération 3D avec Nerfstudio
    """
    try:
        jobs_status[job_id]["status"] = "processing"
        jobs_status[job_id]["message"] = "Extraction des frames..."
        jobs_status[job_id]["progress"] = 10
        
        # Créer les dossiers pour ce job
        job_dir = JOBS_DIR / job_id
        frames_dir = job_dir / "images"
        output_dir = job_dir / "output"
        
        frames_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Étape 1: Extraire les frames
        num_frames = extract_frames_from_video(video_path, frames_dir, target_frames=50)
        
        if num_frames < 10:
            raise Exception(f"Pas assez de frames extraites ({num_frames}). Vidéo trop courte?")
        
        jobs_status[job_id]["progress"] = 20
        jobs_status[job_id]["message"] = "Traitement COLMAP (structure from motion)..."
        
        # Étape 2: COLMAP pour estimer les poses de caméra
        colmap_cmd = [
            "ns-process-data", "video",
            "--data", str(frames_dir),
            "--output-dir", str(job_dir / "colmap"),
            "--num-frames-target", str(num_frames)
        ]
        
        print(f"🔄 Exécution COLMAP: {' '.join(colmap_cmd)}")
        
        process = await asyncio.create_subprocess_exec(
            *colmap_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "COLMAP failed"
            print(f"❌ COLMAP Error: {error_msg}")
            raise Exception(f"COLMAP processing failed: {error_msg}")
        
        jobs_status[job_id]["progress"] = 40
        jobs_status[job_id]["message"] = "Entraînement du modèle NeRF (Instant-NGP)..."
        
        # Étape 3: Entraînement Instant-NGP avec Nerfstudio
        train_cmd = [
            "ns-train", "instant-ngp",
            "--data", str(job_dir / "colmap"),
            "--output-dir", str(output_dir),
            "--max-num-iterations", "10000",  # ~2-3 minutes sur RTX 4090
            "--pipeline.model.predict-normals", "True",
            "--vis", "viewer+tensorboard"
        ]
        
        print(f"🔄 Entraînement NeRF: {' '.join(train_cmd)}")
        
        process = await asyncio.create_subprocess_exec(
            *train_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Simuler la progression pendant l'entraînement
        progress = 40
        while process.returncode is None:
            await asyncio.sleep(5)
            progress = min(progress + 5, 85)
            jobs_status[job_id]["progress"] = progress
            
            try:
                await asyncio.wait_for(process.wait(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Training failed"
            print(f"❌ Training Error: {error_msg}")
            raise Exception(f"NeRF training failed: {error_msg}")
        
        jobs_status[job_id]["progress"] = 90
        jobs_status[job_id]["message"] = "Export du modèle 3D..."
        
        # Étape 4: Export en PLY
        # Trouver le dernier checkpoint
        checkpoint_dir = output_dir / "instant-ngp" / "nerfstudio_models"
        checkpoints = list(checkpoint_dir.glob("step-*.ckpt"))
        
        if not checkpoints:
            raise Exception("Aucun checkpoint trouvé après l'entraînement")
        
        latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        export_cmd = [
            "ns-export", "pointcloud",
            "--load-config", str(output_dir / "instant-ngp" / "nerfstudio_models" / "config.yml"),
            "--output-dir", str(OUTPUT_DIR),
            "--num-points", "1000000",  # 1M points pour qualité professionnelle
            "--remove-outliers", "True",
            "--normal-method", "model_output"
        ]
        
        print(f"🔄 Export PLY: {' '.join(export_cmd)}")
        
        process = await asyncio.create_subprocess_exec(
            *export_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Export failed"
            print(f"❌ Export Error: {error_msg}")
            raise Exception(f"PLY export failed: {error_msg}")
        
        # Trouver le fichier PLY généré
        ply_files = list(OUTPUT_DIR.glob(f"{job_id}*.ply"))
        
        if not ply_files:
            raise Exception("Fichier PLY non trouvé après l'export")
        
        ply_file = ply_files[0]
        final_ply = OUTPUT_DIR / f"{job_id}.ply"
        
        if ply_file != final_ply:
            shutil.move(str(ply_file), str(final_ply))
        
        # Succès !
        jobs_status[job_id]["status"] = "completed"
        jobs_status[job_id]["progress"] = 100
        jobs_status[job_id]["message"] = "Modèle 3D généré avec succès"
        jobs_status[job_id]["download_url"] = f"/download/{job_id}.ply"
        
        print(f"✅ Job {job_id} terminé avec succès")
        
        # Nettoyer les fichiers temporaires (garder seulement le PLY final)
        shutil.rmtree(job_dir, ignore_errors=True)
        if video_path.exists():
            video_path.unlink()
        
    except Exception as e:
        print(f"❌ Erreur job {job_id}: {str(e)}")
        jobs_status[job_id]["status"] = "failed"
        jobs_status[job_id]["message"] = "Échec de la génération"
        jobs_status[job_id]["error"] = str(e)
        jobs_status[job_id]["progress"] = 0


@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "online",
        "service": "3D Generation API",
        "version": "1.0.0",
        "gpu": "RTX 4090",
        "technology": "Nerfstudio Instant-NGP"
    }


@app.post("/generate-3d")
async def generate_3d(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Endpoint principal: Upload vidéo et démarre la génération 3D
    
    Returns:
        job_id pour suivre la progression
    """
    try:
        # Vérifier le type de fichier
        if not file.content_type.startswith("video/"):
            raise HTTPException(400, "Le fichier doit être une vidéo")
        
        # Générer un job_id unique
        job_id = str(uuid.uuid4())
        
        # Sauvegarder la vidéo
        video_path = UPLOAD_DIR / f"{job_id}.mp4"
        
        with open(video_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        file_size_mb = len(content) / (1024 * 1024)
        print(f"📥 Vidéo reçue: {file_size_mb:.2f} MB")
        
        # Initialiser le statut du job
        jobs_status[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "message": "En attente de traitement...",
            "progress": 0,
            "created_at": datetime.now().isoformat(),
            "download_url": None,
            "error": None
        }
        
        # Lancer le traitement en arrière-plan
        background_tasks.add_task(process_3d_generation, job_id, video_path)
        
        return {
            "success": True,
            "job_id": job_id,
            "message": "Génération démarrée",
            "estimated_time": "2-5 minutes"
        }
        
    except Exception as e:
        print(f"❌ Erreur upload: {str(e)}")
        raise HTTPException(500, f"Erreur lors de l'upload: {str(e)}")


@app.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    """
    Récupère le statut d'un job de génération
    """
    if job_id not in jobs_status:
        raise HTTPException(404, "Job non trouvé")
    
    return jobs_status[job_id]


@app.get("/download/{filename}")
async def download_model(filename: str):
    """
    Télécharge un modèle 3D généré
    """
    file_path = OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(404, "Fichier non trouvé")
    
    return FileResponse(
        path=str(file_path),
        media_type="application/octet-stream",
        filename=filename
    )


@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """
    Supprime un job et ses fichiers associés
    """
    if job_id in jobs_status:
        # Supprimer les fichiers
        ply_file = OUTPUT_DIR / f"{job_id}.ply"
        if ply_file.exists():
            ply_file.unlink()
        
        # Supprimer du statut
        del jobs_status[job_id]
        
        return {"success": True, "message": "Job supprimé"}
    
    raise HTTPException(404, "Job non trouvé")


@app.get("/jobs")
async def list_jobs():
    """
    Liste tous les jobs
    """
    return {
        "jobs": list(jobs_status.values()),
        "total": len(jobs_status)
    }


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Démarrage du serveur 3D Generation API...")
    print("📍 URL: http://0.0.0.0:8000")
    print("🎮 GPU: RTX 4090")
    print("🔬 Tech: Nerfstudio Instant-NGP")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
