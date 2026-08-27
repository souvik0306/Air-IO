#!/bin/bash

set -euo pipefail

REPO_DIR="${AIRIO_REPO_DIR:-$HOME/Air-IO}"
IMAGE="${AIRIO_IMAGE:-/app1/common/singularity-img/vanda/pytorch_2.5_cuda_12.4_unsloth.sif}"

cd "$REPO_DIR"

echo "========================================"
echo "Air-IO setup"
echo "========================================"

module load apptainer/1.4.1

download_and_extract() {
    local directory="$1"
    local url="$2"
    local archive="${url##*/}"

    if [ -d "$directory" ]; then
        echo "$directory already exists"
        return
    fi

    wget -c "$url" -O "$archive"
    unzip -q "$archive"
}

echo "Downloading TLab datasets"
download_and_extract \
    "T-Lab_31st_July_dataset" \
    "https://github.com/souvik0306/AirIMU/releases/download/31st_July_fast_agile/T-Lab_31st_July_dataset.zip"
download_and_extract \
    "T-Lab_28th_July_dataset" \
    "https://github.com/souvik0306/AirIMU/releases/download/28th_july_hover/T-Lab_28th_July_dataset.zip"

echo "Downloading pretrained Air-IO EuRoC weights"
download_and_extract \
    "AirIO_EuRoC" \
    "https://github.com/Air-IO/Air-IO/releases/download/AirIO/AirIO_EuRoC.zip"

echo "Installing Python requirements"
apptainer exec -e "$IMAGE" \
    python3 -m pip install --user -r requirements.txt

checkpoint="./AirIO_EuRoC/AirIO_checkpoint/best_model.ckpt"
if [ ! -f "$checkpoint" ]; then
    echo "ERROR: Air-IO checkpoint not found at $checkpoint"
    exit 1
fi

for dataset in T-Lab_31st_July_dataset T-Lab_28th_July_dataset; do
    if [ ! -d "./$dataset" ]; then
        echo "ERROR: dataset not found: $dataset"
        exit 1
    fi
done

echo "Checking PyTorch and CUDA"
apptainer exec --nv -e "$IMAGE" python3 - <<'PYTHON'
import torch

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
PYTHON

echo "========================================"
echo "Air-IO setup complete"
echo "========================================"
