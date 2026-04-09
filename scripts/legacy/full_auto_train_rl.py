"""
full_auto_train_rl.py — Hệ thống Tự Trị Toàn Phần cho PPO Phase 2
==================================================================
1. Chờ instance sẵn sàng
2. Deploy code + data + BC checkpoint lên server
3. Ép PyTorch cu118 (fix driver cũ)
4. Khởi chạy PPO RL Training trong tmux
5. Guardian canh gác, tự hút model về khi xong
6. Tự hủy server
"""

import subprocess
import time
import json
import os

INSTANCE_ID = "34406628"
DATASET = "data/processed/XAUUSD_M5_w256.h5"
BC_CKPT = "checkpoints/best_model_bc.pt"
BASH = r"C:\Program Files\Git\bin\bash.exe"

def run_bash(cmd, capture=False):
    return subprocess.run([BASH, "-c", cmd], capture_output=capture, text=capture)

def get_ssh_info():
    res = subprocess.run(["vastai", "ssh-url", INSTANCE_ID], capture_output=True, text=True)
    if res.returncode != 0: return None
    url = res.stdout.strip().replace("ssh://", "")
    parts = url.split(":")
    if len(parts) != 2: return None
    host = parts[0].split("@")[-1]
    return host, parts[1]

def wait_for_instance():
    print("⏳ Chờ instance khởi động...")
    for i in range(60):
        res = subprocess.run(["vastai", "show", "instances", "--raw"], capture_output=True, text=True)
        try:
            data = json.loads(res.stdout)
            inst = next((x for x in data if str(x["id"]) == INSTANCE_ID), None)
            if inst and inst.get("actual_status") == "running":
                print(f"✅ Instance {INSTANCE_ID} đã sẵn sàng!")
                return True
        except: pass
        print(f"   Lần {i+1}/60... chờ 15s")
        time.sleep(15)
    return False

def deploy_code(host, port):
    print("📦 Uploading source code...")
    run_bash(f"scp -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -P {port} -r src scripts Dockerfile requirements.txt root@{host}:/workspace/")
    
    print("📦 Uploading HDF5 dataset...")
    run_bash(f"ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -p {port} root@{host} 'mkdir -p /workspace/data/processed /workspace/logs /workspace/checkpoints'")
    run_bash(f"scp -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -P {port} {DATASET} root@{host}:/workspace/data/processed/")
    
    print("📦 Uploading BC Checkpoint...")
    run_bash(f"scp -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -P {port} {BC_CKPT} root@{host}:/workspace/checkpoints/best_model_bc.pt")
    
    print("🔧 Fix CRLF + dependencies...")
    run_bash(f"ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -p {port} root@{host} 'apt-get update && apt-get install -y dos2unix tmux && cd /workspace && find . -type f \\( -name \"*.sh\" -o -name \"*.py\" \\) -exec dos2unix {{}} + && pip install -q -r requirements.txt'")

def force_cuda(host, port):
    print("🔥 Ép cài PyTorch CUDA 11.8...")
    run_bash(f"ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -p {port} root@{host} 'pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118'")
    print("✅ PyTorch cu118 cài xong!")

def start_rl_training(host, port):
    print("🚀 Khởi chạy PPO RL Training trong tmux...")
    cmd = "nohup python src/training/train_rl.py --h5 data/processed/XAUUSD_M5_w256.h5 --bc-ckpt checkpoints/best_model_bc.pt --n-envs 64 --total-steps 2000000 > logs/train_rl.log 2>&1"
    run_bash(f"ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -p {port} root@{host} \"cd /workspace && tmux new-session -d -s train_session 'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 && {cmd}'\"")
    print("✅ PPO RL Training đã khởi chạy!")

def verify_cuda(host, port):
    time.sleep(20)
    print("🔍 Kiểm tra Device...")
    res = run_bash(f"ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -p {port} root@{host} 'cat /workspace/logs/train_rl.log'", capture=True)
    if "Device: cuda" in res.stdout:
        print("✅ XANH LÁ — GPU CUDA kích hoạt!")
        return True
    elif "Device: cpu" in res.stdout:
        print("🔴 ĐỎ — Bị đẩy về CPU!")
        return False
    else:
        print("⚠️  Chưa thấy log, chờ thêm 20s...")
        time.sleep(20)
        res = run_bash(f"ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -p {port} root@{host} 'cat /workspace/logs/train_rl.log'", capture=True)
        print(res.stdout[-800:] if len(res.stdout) > 800 else res.stdout)
        return "Device: cuda" in res.stdout

def guardian_loop(host, port):
    print("\n🛡️  GUARDIAN RL ACTIVATED — Canh gác mỗi 5 phút...")
    print(f"   Server: {host}:{port}")
    print(f"   Dự kiến: 2-4 tiếng cho PPO 2M steps\n")
    
    while True:
        res = run_bash(f"ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -p {port} root@{host} pgrep -f train_rl.py", capture=True)
        
        if res.returncode == 255:
            print("[!] Lỗi SSH. Retry sau 5 phút...")
            time.sleep(300)
            continue
        
        if res.returncode == 1 or not res.stdout.strip():
            print("\n🎉 PPO RL TRAINING ĐÃ HOÀN THÀNH!")
            
            print("[+] Thu hoạch Checkpoints...")
            os.makedirs("checkpoints", exist_ok=True)
            run_bash(f"scp -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -P {port} -r root@{host}:/workspace/checkpoints/* checkpoints/")
            
            print("[+] Thu hoạch Logs...")
            os.makedirs("logs", exist_ok=True)
            run_bash(f"scp -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -P {port} -r root@{host}:/workspace/logs/* logs/")
            
            for f in ["ppo_best.pt", "ppo_final.pt"]:
                path = f"checkpoints/{f}"
                if os.path.exists(path):
                    size_mb = os.path.getsize(path) / 1024 / 1024
                    print(f"✅ {f} đã về máy! ({size_mb:.1f} MB)")
            
            print("[+] Kết liễu Server...")
            subprocess.run(["vastai", "destroy", "instance", INSTANCE_ID])
            
            print("\n" + "="*50)
            print("🏆 PPO RL TRAINING HOÀN TẤT!")
            print("   Model PPO M5_w256 nằm trong checkpoints/")
            print("   Server đã hủy.")
            print("="*50)
            break
        else:
            pids = res.stdout.strip().replace('\n', ', ')
            log_res = run_bash(f"ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -p {port} root@{host} 'tail -3 /workspace/logs/train_rl.log'", capture=True)
            last_line = log_res.stdout.strip().split('\n')[-1] if log_res.stdout.strip() else "..."
            print(f"[⏳] PPO vẫn đang cày (PIDs: {pids})")
            print(f"     {last_line}")
            print(f"     Ngủ 5 phút...\n")
            time.sleep(300)


if __name__ == "__main__":
    print("=" * 60)
    print("🤖 SCALPEX200 — PPO RL PHASE 2")
    print(f"   Instance: {INSTANCE_ID}")
    print(f"   Dataset:  {DATASET}")
    print(f"   BC Ckpt:  {BC_CKPT}")
    print("=" * 60)
    
    if not wait_for_instance():
        print("❌ Timeout!"); exit(1)
    
    ssh = get_ssh_info()
    if not ssh:
        print("❌ SSH fail!"); exit(1)
    host, port = ssh
    print(f"🔗 SSH: root@{host}:{port}")
    
    deploy_code(host, port)
    force_cuda(host, port)
    start_rl_training(host, port)
    
    if verify_cuda(host, port):
        guardian_loop(host, port)
    else:
        print("⚠️  Chạy trên CPU, Guardian vẫn tiếp tục...")
        guardian_loop(host, port)
