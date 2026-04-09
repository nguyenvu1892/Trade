#!/bin/bash
# pull_results.sh — Lấy Checkpoint best_model.pt và XÓA (Trảm) Instance
set -e
INSTANCE_ID=${1:?"Usage: $0 <instance_id> <ssh_key>"}
SSH_KEY=${2:?"Usage: $0 <instance_id> <ssh_key>"}

CONN=$(vastai ssh-url $INSTANCE_ID)
SSH_HOST=$(echo $CONN | sed 's/ssh:\/\///')

echo "Pulling checkpoints and logs from Vast.ai..."
rsync -avz -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" root@${SSH_HOST}:/workspace/checkpoints/ ./checkpoints/
rsync -avz -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" root@${SSH_HOST}:/workspace/logs/ ./logs/

echo "✅ Done! Checkpoints saved to ./checkpoints/"

echo ""
echo "⚠️  CẢNH BÁO TIÊU DIỆT TÀN DƯ (TRẢM QUYẾT) ⚠️"
echo "Theo quy trình, bạn tuyệt đối không được giữ lại mạng lưới sau khi train xong."
echo "Bạn có muốn chạy lệnh tiêu diệt: vastai destroy instance $INSTANCE_ID không? (y/n)"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    vastai destroy instance $INSTANCE_ID
    echo "💀 Đã xóa bỏ VĨNH VIỄN Instance để cắt đứt mọi vòi Phí Thuê Máy!"
else
    echo "Hãy Tự xóa bỏ khi bạn xong để tránh tốn kém!"
fi
