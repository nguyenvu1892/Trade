#!/bin/bash
# vast_launch.sh — Triển khai đào tạo (BC / PPO) lên Vast.ai instance
# Đã Tối Ưu Hóa theo Quy trình NOHUP CRASH-PROOF & Giao Thức Nhảy Cóc Băng Thông

set -e

INSTANCE_ID=${1:?"Usage: $0 <instance_id> <ssh_key> <mode: bc|rl> <dataset_h5> [bc_ckpt]"}
SSH_KEY=${2:?"Usage: $0 <instance_id> <ssh_key> <mode: bc|rl> <dataset_h5> [bc_ckpt]"}
MODE=${3:?"Mode must be 'bc' or 'rl'"}
DATASET_H5=${4:?"Path to H5 dataset"}
BC_CKPT=${5:-"checkpoints/best_model_bc.pt"}

if [ "$MODE" != "bc" ] && [ "$MODE" != "rl" ]; then
    echo "Lỗi: Mode phải là 'bc' (Behavioral Cloning) hoặc 'rl' (PPO)"
    exit 1
fi

DATASET_NAME=$(basename "$DATASET_H5")

echo "=== XAUUSD Bot — Vast.ai Deployment ==="
echo "Instance: $INSTANCE_ID"
echo "Mode: $MODE"
echo "Dataset: $DATASET_H5"

# 1. Lấy connection info
CONN=$(vastai ssh-url $INSTANCE_ID)
RAW_URL=$(echo $CONN | sed 's/ssh:\/\///')
SSH_PORT=$(echo $RAW_URL | cut -d':' -f2)
SSH_HOST=$(echo $RAW_URL | cut -d':' -f1 | cut -d'@' -f2)

# 2. Upload source code (KHÔNG BƠM DATA RAW LÊN CLOUD)
echo "Uploading source code via SCP..."
scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -P "$SSH_PORT" -r src scripts Dockerfile requirements.txt root@${SSH_HOST}:/workspace/

# Bơm chính xác File HDF5 Tinh Luyện (Giao Thức Nhảy Cóc Băng Thông)
echo "Uploading Processed HDF5 Database..."
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -p "$SSH_PORT" root@${SSH_HOST} "mkdir -p /workspace/data/processed"
scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -P "$SSH_PORT" "$DATASET_H5" root@${SSH_HOST}:/workspace/data/processed/

# Upload BC Checkpoint (Chỉ cần khi chạy RL)
if [ "$MODE" == "rl" ] && [ -f "$BC_CKPT" ]; then
    echo "Uploading BC Checkpoint for RL..."
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -p "$SSH_PORT" root@${SSH_HOST} "mkdir -p /workspace/checkpoints"
    scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -P "$SSH_PORT" "$BC_CKPT" root@${SSH_HOST}:/workspace/checkpoints/best_model_bc.pt
fi

# 3. Cài đặt dependencies & Diệt trừ mã độc CRLF của Windows
echo "Fixing CRLF (dos2unix) and installing dependencies..."
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -p "$SSH_PORT" root@${SSH_HOST} \
    "cd /workspace && \
     apt-get update && apt-get install -y dos2unix tmux && \
     find . -type f \( -name '*.sh' -o -name '*.py' \) -exec dos2unix {} + && \
     pip install -q -r requirements.txt"

# Thiết lập lệnh chạy dựa trên Mode
if [ "$MODE" == "bc" ]; then
    CMD="nohup python src/training/train_bc.py --h5 data/processed/$DATASET_NAME --epochs 100 > logs/train_bc.log 2>&1"
    LOG_FILE="train_bc.log"
else
    CMD="nohup python src/training/train_rl.py --h5 data/processed/$DATASET_NAME --bc-ckpt checkpoints/best_model_bc.pt --n-envs 64 --total-steps 2000000 > logs/train_rl.log 2>&1"
    LOG_FILE="train_rl.log"
fi

# 4. Phóng hỏa bằng đai sinh tồn tmux + nohup
echo "Starting $MODE training inside tmux & nohup..."
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -p "$SSH_PORT" root@${SSH_HOST} \
    "cd /workspace && mkdir -p logs && \
     tmux new-session -d -s train_session 'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 && \
        $CMD'"

echo "✅ $MODE training started dynamically via TMUX!"
echo "   Sử dụng lệnh sau để xem log thực tế:"
echo "   ssh -i $SSH_KEY -o StrictHostKeyChecking=no -p $SSH_PORT root@${SSH_HOST} 'tail -f /workspace/logs/$LOG_FILE'"
echo "   Để vào tmux session: ssh ... 'tmux attach -t train_session'"
