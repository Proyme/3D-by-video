"""
Script de test pour l'API de génération 3D
"""
import requests
import time
import sys
from pathlib import Path

# Configuration
BACKEND_URL = "http://localhost:8000"
TEST_VIDEO = "test_video.mp4"  # Remplacer par votre vidéo de test


def test_health_check():
    """Test 1: Vérifier que l'API est en ligne"""
    print("🔍 Test 1: Health Check")
    try:
        response = requests.get(f"{BACKEND_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API en ligne: {data['service']}")
            print(f"   Version: {data['version']}")
            print(f"   GPU: {data['gpu']}")
            print(f"   Tech: {data['technology']}")
            return True
        else:
            print(f"❌ Erreur: Status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur de connexion: {e}")
        return False


def test_upload_video(video_path: str):
    """Test 2: Upload d'une vidéo et démarrage de la génération"""
    print(f"\n🔍 Test 2: Upload vidéo ({video_path})")
    
    if not Path(video_path).exists():
        print(f"❌ Fichier non trouvé: {video_path}")
        return None
    
    try:
        with open(video_path, 'rb') as f:
            files = {'file': (video_path, f, 'video/mp4')}
            response = requests.post(f"{BACKEND_URL}/generate-3d", files=files)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                job_id = data['job_id']
                print(f"✅ Upload réussi")
                print(f"   Job ID: {job_id}")
                print(f"   Temps estimé: {data.get('estimated_time', 'N/A')}")
                return job_id
            else:
                print(f"❌ Échec: {data}")
                return None
        else:
            print(f"❌ Erreur HTTP {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None


def test_job_status(job_id: str, max_wait: int = 600):
    """Test 3: Suivre le statut d'un job jusqu'à la fin"""
    print(f"\n🔍 Test 3: Suivi du job {job_id}")
    
    start_time = time.time()
    last_progress = -1
    
    while True:
        try:
            response = requests.get(f"{BACKEND_URL}/job-status/{job_id}")
            
            if response.status_code == 404:
                print(f"❌ Job non trouvé: {job_id}")
                return None
            
            if response.status_code == 200:
                data = response.json()
                status = data['status']
                progress = data.get('progress', 0)
                message = data.get('message', '')
                
                # Afficher seulement si le progrès a changé
                if progress != last_progress:
                    elapsed = int(time.time() - start_time)
                    print(f"   [{elapsed}s] {status.upper()}: {progress}% - {message}")
                    last_progress = progress
                
                if status == 'completed':
                    download_url = data.get('download_url')
                    print(f"✅ Génération terminée !")
                    print(f"   Temps total: {elapsed}s")
                    print(f"   URL de téléchargement: {download_url}")
                    return download_url
                
                elif status == 'failed':
                    error = data.get('error', 'Erreur inconnue')
                    print(f"❌ Génération échouée: {error}")
                    return None
                
                # Timeout
                if time.time() - start_time > max_wait:
                    print(f"⏱️ Timeout après {max_wait}s")
                    return None
                
                # Attendre avant le prochain poll
                time.sleep(5)
            else:
                print(f"❌ Erreur HTTP {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return None


def test_download_model(download_url: str, output_path: str = "output_model.ply"):
    """Test 4: Télécharger le modèle 3D généré"""
    print(f"\n🔍 Test 4: Téléchargement du modèle")
    
    try:
        full_url = f"{BACKEND_URL}{download_url}"
        response = requests.get(full_url, stream=True)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            file_size = Path(output_path).stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            
            print(f"✅ Modèle téléchargé: {output_path}")
            print(f"   Taille: {file_size_mb:.2f} MB")
            return True
        else:
            print(f"❌ Erreur HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False


def test_list_jobs():
    """Test 5: Lister tous les jobs"""
    print(f"\n🔍 Test 5: Liste des jobs")
    
    try:
        response = requests.get(f"{BACKEND_URL}/jobs")
        
        if response.status_code == 200:
            data = response.json()
            total = data.get('total', 0)
            jobs = data.get('jobs', [])
            
            print(f"✅ {total} job(s) trouvé(s)")
            
            for job in jobs[:5]:  # Afficher max 5 jobs
                print(f"   - {job['job_id']}: {job['status']} ({job.get('progress', 0)}%)")
            
            return True
        else:
            print(f"❌ Erreur HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False


def run_full_test(video_path: str):
    """Exécuter tous les tests"""
    print("=" * 60)
    print("🧪 TESTS API DE GÉNÉRATION 3D")
    print("=" * 60)
    
    # Test 1: Health check
    if not test_health_check():
        print("\n❌ API non accessible. Arrêt des tests.")
        return False
    
    # Test 2: Upload vidéo
    job_id = test_upload_video(video_path)
    if not job_id:
        print("\n❌ Upload échoué. Arrêt des tests.")
        return False
    
    # Test 3: Suivre le job
    download_url = test_job_status(job_id, max_wait=600)
    if not download_url:
        print("\n❌ Génération échouée. Arrêt des tests.")
        return False
    
    # Test 4: Télécharger le modèle
    if not test_download_model(download_url):
        print("\n❌ Téléchargement échoué.")
        return False
    
    # Test 5: Lister les jobs
    test_list_jobs()
    
    print("\n" + "=" * 60)
    print("✅ TOUS LES TESTS RÉUSSIS !")
    print("=" * 60)
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = TEST_VIDEO
    
    print(f"📹 Vidéo de test: {video_path}")
    print(f"🌐 Backend URL: {BACKEND_URL}")
    print()
    
    success = run_full_test(video_path)
    sys.exit(0 if success else 1)
