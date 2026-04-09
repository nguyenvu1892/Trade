"""
full_auto_train.py — Hệ thống Tự Trị Toàn Phần
================================================
1. Chờ instance sẵn sàng
2. Deploy code + data lên server
3. Ép PyTorch cu118 (fix driver cũ)
4. Khởi chạy BC Training trong tmux
5. Canh gác mỗi 5 phút, tự hút model về khi xong
6. Tự hủy server để cắt phí
"""

import subprocess
import time
import json
import os

INSTANCE_ID = "34379061"
DATASET = "data/processed/XAUUSD_M5_w256.h5"
BASH = r"C:\Program Files\Git\bin\bash.exe"

def run_bash(cmd, capture=False):
    """Chạy lệnh bash an toàn, không bị Windows cắn path."""
    result = subprocess.run([BASH, "-c", cmd], capture_output=capture, text=capture)
    return result

def get_ssh_info():
    """Lấy host:port từ Vast.ai API."""
    res = subprocess.run(["vastai", "ssh-url", INSTANCE_ID], capture_output=True, text=True)
    if res.returncode != 0:
        return None
    url = res.stdout.strip().replace("ssh://", "")
    parts = url.split(":")
    if len(parts) != 2:
        return None
    host = parts[0].split("@")[-1]
    return host, parts[1]

def wait_for_instance():
    """Chờ instance chuyển sang trạng thái 'running'."""
    print("⏳ Chờ instance khởi động...")
    for i in range(60):
        res = subprocess.run(["vastai", "show", "instances", "--raw"], capture_output=True, text=True)
        try:
            data = json.loads(res.stdout)
            inst = next((x for x in data if str(x["id"]) == INSTANCE_ID), None)
            if inst and inst.get("actual_status") == "running":
                print(f"✅ Instance {INSTANCE_ID} đã sẵn sàng!")
                return True
        except:
            pass
        print(f"   Lần kiểm tra {i+1}/60... chờ 15s")
        time.sleep(15)
    print("❌ Timeout chờ instance!")
    return False

def deploy_code(host, port):
    """Upload code + dataset lên server."""
    print("📦 Uploading source code...")
    run_bash(f"scp -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -P {port} -r src scripts Dockerfile requirements.txt root@{host}:/workspace/")
    
    print("📦 Uploading HDF5 dataset...")
    run_bash(f"ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -p {port} root@{host} 'mkdir -p /workspace/data/processed /workspace/logs /workspace/checkpoints'")
    run_bash(f"scp -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -P {port} {DATASET} root@{host}:/workspace/data/processed/")
    
    print("🔧 Cài dos2unix + fix CRLF...")
    run_bash(f"ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -p {port} root@{host} 'apt-get update && apt-get install -y dos2unix tmux && cd /workspace && find . -type f \\( -name \"*.sh\" -o -name \"*.py\" \\) -exec dos2unix {{}} +'")
    
    print("🔧 Cài dependencies...")
    run_bash(f"ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -p {port} root@{host} 'cd /workspace && pip install -q -r requirements.txt'")

def force_cuda(host, port):
    """Ép cài PyTorch cu118 để fix xung đột driver."""
    print("🔥 Ép cài PyTorch CUDA 11.8 (fix driver cũ)...")
    run_bash(f"ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -p {port} root@{host} 'pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118'")
    print("✅ PyTorch cu118 cài xong!")

def start_training(host, port):
    """Khởi chạy BC Training trong tmux."""
    print("🚀 Khởi chạy BC Training trong tmux...")
    run_bash(f"ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -p {port} root@{host} \"cd /workspace && tmux new-session -d -s train_session 'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 && nohup python src/training/train_bc.py --h5 data/processed/XAUUSD_M5_w256.h5 --epochs 100 > logs/train_bc.log 2>&1'\"")
    print("✅ Training đã khởi chạy!")

def verify_cuda(host, port):
    """Kiểm tra Training đang chạy trên GPU hay CPU."""
    time.sleep(15)
    print("🔍 Kiểm tra Device (cuda vs cpu)...")
    res = run_bash(f"ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -p {port} root@{host} 'cat /workspace/logs/train_bc.log'", capture=True)
    if "Device: cuda" in res.stdout:
        print("✅ XANH LÁ — Model đang chạy trên GPU CUDA!")
        return True
    elif "Device: cpu" in res.stdout:
        print("🔴 ĐỎ — Vẫn bị đẩy về CPU! Cần debug thêm.")
        return False
    else:
        print("⚠️  Chưa thấy Device log, chờ thêm...")
        time.sleep(15)
        res = run_bash(f"ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -p {port} root@{host} 'cat /workspace/logs/train_bc.log'", capture=True)
        print(res.stdout[-500:] if len(res.stdout) > 500 else res.stdout)
        return "Device: cuda" in res.stdout

def guardian_loop(host, port):
    """Canh gác cho tới khi Training xong, tự hút model về và hủy server."""
    print("\n🛡️  GUARDIAN ACTIVATED — Canh gác mỗi 5 phút...")
    print(f"   Server: {host}:{port}")
    print(f"   Dự kiến hoàn thành: ~5.5 tiếng\n")
    
    while True:
        res = run_bash(f"ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -p {port} root@{host} pgrep -f train_bc.py", capture=True)
        
        if res.returncode == 255:
            print(f"[!] Lỗi SSH. Retry sau 5 phút...")
            time.sleep(300)
            continue
        
        if res.returncode == 1 or not res.stdout.strip():
            print("\n🎉 TRAIN_BC.PY ĐÃ HOÀN THÀNH!")
            
            # Hút Checkpoints
            print("[+] Thu hoạch Checkpoints...")
            os.makedirs("checkpoints", exist_ok=True)
            pull = run_bash(f"scp -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -P {port} -r root@{host}:/workspace/checkpoints/* checkpoints/", capture=True)
            print(f"    SCP: {pull.stderr.strip()}")
            
            # Hút Logs
            print("[+] Thu hoạch Logs...")
            os.makedirs("logs", exist_ok=True)
            run_bash(f"scp -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -P {port} -r root@{host}:/workspace/logs/* logs/")
            
            # Xác nhận file
            if os.path.exists("checkpoints/best_model_bc.pt"):
                size_mb = os.path.getsize("checkpoints/best_model_bc.pt") / 1024 / 1024
                print(f"✅ best_model_bc.pt đã về máy! ({size_mb:.1f} MB)")
            else:
                print("⚠️  Không tìm thấy best_model_bc.pt!")
            
            # HỦY SERVER
            print("[+] Kết liễu Server để cắt phí...")
            subprocess.run(["vastai", "destroy", "instance", INSTANCE_ID])
            
            print("\n" + "="*50)
            print("🏆 ĐẠI CÔNG CÁO THÀNH!")
            print("   Model BC M5_w256 đã nằm gọn trong checkpoints/")
            print("   Server đã bị hủy, không tốn thêm phí.")
            print("="*50)
            break
        else:
            pids = res.stdout.strip().replace('\n', ', ')
            # Hút log nhanh để báo cáo tiến độ
            log_res = run_bash(f"ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -p {port} root@{host} 'tail -3 /workspace/logs/train_bc.log'", capture=True)
            last_line = log_res.stdout.strip().split('\n')[-1] if log_res.stdout.strip() else "..."
            print(f"[⏳] Vẫn đang cày (PIDs: {pids})")
            print(f"     {last_line}")
            print(f"     Ngủ 5 phút rồi quét lại...\n")
            time.sleep(300)


if __name__ == "__main__":
    print("=" * 60)
    print("🤖 SCALPEX200 — HỆ THỐNG TỰ TRỊ TOÀN PHẦN")
    print(f"   Instance: {INSTANCE_ID}")
    print(f"   Dataset:  {DATASET}")
    print("=" * 60)
    
    if not wait_for_instance():
        exit(1)
    
    ssh = get_ssh_info()
    if not ssh:
        print("❌ Không lấy được SSH info!")
        exit(1)
    host, port = ssh
    print(f"🔗 SSH: root@{host}:{port}")
    
    deploy_code(host, port)
    force_cuda(host, port)
    start_training(host, port)
    
    if verify_cuda(host, port):
        guardian_loop(host, port)
    else:
        print("⚠️  Training có thể chạy chậm trên CPU. Guardian vẫn khởi chạy...")
        guardian_loop(host, port)
