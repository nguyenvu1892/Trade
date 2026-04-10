"""
deploy_and_train.py
Upload code + data lên Vast.ai instance, chạy BC + PPO training.
"""
import subprocess, time, json, sys, os

INSTANCE_ID = "34463496"
BASH = r"C:\Program Files\Git\bin\bash.exe"

def run(cmd):
    print(f">>> {cmd[:80]}...")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.stdout.strip():
        print(r.stdout[-500:])
    if r.returncode != 0 and r.stderr.strip():
        print(f"ERR: {r.stderr[-300:]}")
    return r

def ssh(cmd):
    return subprocess.run(
        [BASH, "-c", f'ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -p $PORT root@$HOST "{cmd}"'],
        capture_output=True, text=True,
        env={**os.environ, "PORT": PORT, "HOST": HOST}
    )

def scp_up(local, remote):
    subprocess.run(
        [BASH, "-c", f'scp -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -P $PORT -r {local} root@$HOST:{remote}'],
        capture_output=True, text=True,
        env={**os.environ, "PORT": PORT, "HOST": HOST}
    )

def scp_down(remote, local):
    subprocess.run(
        [BASH, "-c", f'scp -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -P $PORT root@$HOST:{remote} {local}'],
        capture_output=True, text=True,
        env={**os.environ, "PORT": PORT, "HOST": HOST}
    )

# Wait for instance
print("Waiting for instance to boot...")
for i in range(60):
    res = subprocess.run(["vastai", "show", "instances", "--raw"], capture_output=True, text=True)
    try:
        data = json.loads(res.stdout)
        inst = next((x for x in data if str(x["id"]) == INSTANCE_ID), None)
        if inst and inst.get("actual_status") == "running":
            print(f"Instance ready! Status: {inst['actual_status']}")
            break
    except:
        pass
    print(f"  Waiting... ({i*10}s)")
    time.sleep(10)
else:
    print("TIMEOUT!")
    sys.exit(1)

# Get SSH info
res = subprocess.run(["vastai", "ssh-url", INSTANCE_ID], capture_output=True, text=True)
url = res.stdout.strip().replace("ssh://", "")
parts = url.split(":")
HOST = parts[0].split("@")[-1]
PORT = parts[1]
print(f"SSH: {HOST}:{PORT}")

# Upload
print("\n=== UPLOADING ===")
ssh("mkdir -p /workspace/src /workspace/data/processed /workspace/checkpoints /workspace/logs")
scp_up("src", "/workspace/")
scp_up("requirements.txt", "/workspace/")
scp_up("data/processed/XAUUSD_M5_w256.h5", "/workspace/data/processed/")
print("Upload done!")

# Install deps
print("\n=== INSTALLING DEPS ===")
r = ssh("cd /workspace && pip install -q h5py scikit-learn gymnasium && pip install -q -r requirements.txt 2>&1 | tail -5")
print(r.stdout[-300:] if r.stdout else "")

# Fix line endings
ssh("apt-get update -qq && apt-get install -y -qq dos2unix && cd /workspace && find . -name '*.py' -exec dos2unix {} + 2>/dev/null")

# Verify CUDA
r = ssh("python -c 'import torch; print(f\"CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}\")'")
print(f"GPU check: {r.stdout.strip()}")

# Train BC (20 epochs on fresh 6-month data)
print("\n=== TRAINING BC (20 epochs) ===")
r = ssh("cd /workspace && python src/training/train_bc.py --h5 data/processed/XAUUSD_M5_w256.h5 --epochs 20 --window-size 256 2>&1 | tail -30")
print(r.stdout[-1000:] if r.stdout else "")
print(r.stderr[-500:] if r.stderr else "")

# Train PPO with higher entropy (500K steps)
print("\n=== TRAINING PPO (500K steps, high entropy) ===")
r = ssh("cd /workspace && python src/training/train_rl.py --h5 data/processed/XAUUSD_M5_w256.h5 --bc-ckpt checkpoints/best_model_bc.pt --n-envs 64 --total-steps 500000 2>&1 | tail -30")
print(r.stdout[-1000:] if r.stdout else "")
print(r.stderr[-500:] if r.stderr else "")

# Pull models
print("\n=== PULLING MODELS ===")
os.makedirs("checkpoints/fresh_6month", exist_ok=True)
scp_down("/workspace/checkpoints/best_model_bc.pt", "checkpoints/fresh_6month/best_model_bc.pt")
scp_down("/workspace/checkpoints/ppo_best.pt", "checkpoints/fresh_6month/ppo_best.pt")
scp_down("/workspace/checkpoints/ppo_final.pt", "checkpoints/fresh_6month/ppo_final.pt")

for f in ["best_model_bc.pt", "ppo_best.pt", "ppo_final.pt"]:
    path = f"checkpoints/fresh_6month/{f}"
    if os.path.exists(path):
        size = os.path.getsize(path) / 1024 / 1024
        print(f"  {f}: {size:.1f} MB")
    else:
        print(f"  {f}: MISSING!")

# Destroy instance
print("\n=== DESTROYING INSTANCE ===")
subprocess.run(["vastai", "destroy", "instance", INSTANCE_ID])
print("Done! Instance destroyed.")
