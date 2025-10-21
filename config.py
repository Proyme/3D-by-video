"""
Configuration centralisée pour le backend de génération 3D
"""
import os
from pathlib import Path
from typing import Optional

class Config:
    """Configuration principale de l'application"""
    
    # Serveur
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    WORKERS: int = int(os.getenv("WORKERS", "1"))
    
    # Chemins
    BASE_DIR: Path = Path(__file__).parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    OUTPUT_DIR: Path = BASE_DIR / "outputs"
    JOBS_DIR: Path = BASE_DIR / "jobs"
    TEMP_DIR: Path = BASE_DIR / "temp"
    
    # Upload
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "500"))
    MAX_UPLOAD_SIZE_BYTES: int = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    ALLOWED_VIDEO_FORMATS: list = ["mp4", "mov", "avi"]
    
    # Processing
    MAX_CONCURRENT_JOBS: int = int(os.getenv("MAX_CONCURRENT_JOBS", "2"))
    CLEANUP_TEMP_FILES: bool = os.getenv("CLEANUP_TEMP_FILES", "true").lower() == "true"
    
    # Gaussian Splatting
    GAUSSIAN_SPLATTING_PATH: Path = Path(os.getenv(
        "GAUSSIAN_SPLATTING_PATH", 
        "/workspace/gaussian-splatting"
    ))
    ITERATIONS: int = int(os.getenv("ITERATIONS", "7000"))
    DENSIFY_UNTIL_ITER: int = int(os.getenv("DENSIFY_UNTIL_ITER", "5000"))
    DENSIFICATION_INTERVAL: int = int(os.getenv("DENSIFICATION_INTERVAL", "100"))
    
    # Nerfstudio
    MAX_NUM_ITERATIONS: int = int(os.getenv("MAX_NUM_ITERATIONS", "10000"))
    NUM_FRAMES_TARGET: int = int(os.getenv("NUM_FRAMES_TARGET", "50"))
    
    # Sécurité
    API_KEY: Optional[str] = os.getenv("API_KEY")
    ENABLE_AUTH: bool = os.getenv("ENABLE_AUTH", "false").lower() == "true"
    
    # Monitoring
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "false").lower() == "true"
    
    # RunPod (optionnel)
    RUNPOD_API_KEY: Optional[str] = os.getenv("RUNPOD_API_KEY")
    RUNPOD_POD_ID: Optional[str] = os.getenv("RUNPOD_POD_ID")
    
    @classmethod
    def create_directories(cls):
        """Créer tous les dossiers nécessaires"""
        for dir_path in [cls.UPLOAD_DIR, cls.OUTPUT_DIR, cls.JOBS_DIR, cls.TEMP_DIR]:
            dir_path.mkdir(exist_ok=True, parents=True)
    
    @classmethod
    def validate(cls):
        """Valider la configuration"""
        errors = []
        
        # Vérifier Gaussian Splatting
        if not cls.GAUSSIAN_SPLATTING_PATH.exists():
            errors.append(f"Gaussian Splatting non trouvé: {cls.GAUSSIAN_SPLATTING_PATH}")
        
        # Vérifier l'authentification
        if cls.ENABLE_AUTH and not cls.API_KEY:
            errors.append("ENABLE_AUTH=true mais API_KEY non défini")
        
        if errors:
            raise ValueError("Erreurs de configuration:\n" + "\n".join(errors))
    
    @classmethod
    def print_config(cls):
        """Afficher la configuration (sans les secrets)"""
        print("=" * 60)
        print("CONFIGURATION DU BACKEND")
        print("=" * 60)
        print(f"Serveur: {cls.HOST}:{cls.PORT}")
        print(f"Workers: {cls.WORKERS}")
        print(f"Upload max: {cls.MAX_UPLOAD_SIZE_MB} MB")
        print(f"Formats vidéo: {', '.join(cls.ALLOWED_VIDEO_FORMATS)}")
        print(f"Jobs simultanés: {cls.MAX_CONCURRENT_JOBS}")
        print(f"Gaussian Splatting: {cls.GAUSSIAN_SPLATTING_PATH}")
        print(f"Itérations: {cls.ITERATIONS}")
        print(f"Authentification: {'Activée' if cls.ENABLE_AUTH else 'Désactivée'}")
        print(f"Métriques: {'Activées' if cls.ENABLE_METRICS else 'Désactivées'}")
        print("=" * 60)


# Configuration par défaut
config = Config()

# Créer les dossiers au démarrage
config.create_directories()


if __name__ == "__main__":
    # Test de la configuration
    try:
        config.validate()
        config.print_config()
        print("✅ Configuration valide")
    except ValueError as e:
        print(f"❌ {e}")
