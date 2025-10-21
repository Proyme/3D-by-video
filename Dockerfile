# Dockerfile pour RunPod avec RTX 4090
# Base image avec CUDA 12.1 et PyTorch
FROM runpod/pytorch:2.1.2-py3.10-cuda12.1.1-devel-ubuntu22.04

# Métadonnées
LABEL maintainer="3D Generation API"
LABEL description="Backend FastAPI avec Nerfstudio pour génération 3D"

# Variables d'environnement
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    build-essential \
    cmake \
    ninja-build \
    libopencv-dev \
    libboost-all-dev \
    libeigen3-dev \
    libceres-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Installer COLMAP (requis pour Nerfstudio)
RUN git clone https://github.com/colmap/colmap.git /tmp/colmap && \
    cd /tmp/colmap && \
    git checkout 3.8 && \
    mkdir build && cd build && \
    cmake .. -GNinja \
        -DCMAKE_CUDA_ARCHITECTURES=89 \
        -DCMAKE_BUILD_TYPE=Release && \
    ninja && \
    ninja install && \
    rm -rf /tmp/colmap

# Créer le répertoire de travail
WORKDIR /app

# Copier les requirements
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt

# Installer Nerfstudio avec les extras
RUN pip install --no-cache-dir nerfstudio[gen] && \
    ns-install-cli

# Copier le code de l'application
COPY . .

# Créer les dossiers nécessaires
RUN mkdir -p uploads outputs jobs temp

# Exposer le port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Commande de démarrage
CMD ["python", "main.py"]
