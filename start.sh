#!/bin/bash

# Script de démarrage pour le backend 3D Generation
# À utiliser sur RunPod

echo "🚀 Démarrage du Backend 3D Generation..."
echo "================================================"

# Vérifier CUDA
echo "🔍 Vérification CUDA..."
nvidia-smi

# Vérifier Python
echo "🐍 Version Python:"
python --version

# Vérifier PyTorch et CUDA
echo "🔥 Vérification PyTorch CUDA:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"Aucun\"}')"

# Vérifier Nerfstudio
echo "🎨 Vérification Nerfstudio:"
ns-train --help > /dev/null 2>&1 && echo "✅ Nerfstudio installé" || echo "❌ Nerfstudio non trouvé"

# Créer les dossiers nécessaires
mkdir -p uploads outputs jobs temp

# Nettoyer les anciens fichiers (optionnel)
# rm -rf uploads/* jobs/* temp/*

echo "================================================"
echo "✅ Prêt à démarrer le serveur"
echo "📍 URL: http://0.0.0.0:8000"
echo "🎮 GPU: RTX 4090"
echo "================================================"

# Démarrer le serveur avec uvicorn
# Utiliser --workers 1 car le traitement GPU ne peut pas être parallélisé facilement
uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info \
    --access-log \
    --use-colors
