import subprocess, time, json, sys, os

print("Renting new instance...")
res = subprocess.run(
    ["vastai", "search", "offers", "gpu_name=RTX_4090 num_gpus=1 rentable=True verified=true cpu_ram>=64 disk_space>=100 dph_total<=0.5", "--on-demand", "--raw"],
    capture_output=True, text=True
)

data = json.loads(res.stdout)
best = data[0]
print(f"Renting ID {best['id']} at ${best['dph_total']:.3f}/hr")
res2 = subprocess.run(
    ["vastai", "create", "instance", str(best["id"]), "--image", "pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime", "--disk", "100", "--raw"],
    capture_output=True, text=True
)
try:
    new_id = str(json.loads(res2.stdout)["new_contract"])
except:
    new_id = str(json.loads(res2.stdout.split("{", 1)[1].rsplit("}", 1)[0].join(["{", "}"]))["new_contract"])
    
print(f"New Instance ID: {new_id}")

BASH = r"C:\Program Files\Git\bin\bash.exe"
print("Waiting for boot...")
for i in range(60):
    res = subprocess.run(["vastai", "show", "instances", "--raw"], capture_output=True, text=True)
    try:
        insts = json.loads(res.stdout)
        inst = next((x for x in insts if str(x["id"]) == new_id), None)
        if inst and inst.get("actual_status") == "running":
            print("Ready!")
            break
    except: pass
    time.sleep(5)

res = subprocess.run(["vastai", "ssh-url", new_id], capture_output=True, text=True)
url = res.stdout.strip().replace("ssh://", "")
HOST = url.split("@")[-1].split(":")[0]
PORT = url.split(":")[-1]

def ssh(cmd): return subprocess.run([BASH, "-c", f'ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -p {PORT} root@{HOST} "{cmd}"'], capture_output=True, text=True)
def scp(local, remote): return subprocess.run([BASH, "-c", f'scp -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -P {PORT} -r {local} root@{HOST}:{remote}'], capture_output=True, text=True)
def scp_d(remote, local): return subprocess.run([BASH, "-c", f'scp -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -P {PORT} root@{HOST}:{remote} {local}'], capture_output=True, text=True)

print("Uploading...")
ssh("mkdir -p /workspace/src /workspace/data/processed /workspace/checkpoints /workspace/logs")
scp("src", "/workspace/")
scp("data/processed/XAUUSD_M5_w256.h5", "/workspace/data/processed/")
scp("checkpoints/best_model_bc.pt", "/workspace/checkpoints/")

print("Installing deps (SKIP PyTorch install to keep GPU)...")
ssh("pip install h5py scikit-learn gymnasium pandas numpy stable_baselines3")
ssh("apt-get update -qq && apt-get install -y -qq dos2unix && cd /workspace && find . -name '*.py' -exec dos2unix {} +")

# Verify
r = ssh("python -c \"import torch; print('CUDA:', torch.cuda.is_available())\"")
print(r.stdout.strip())
if "CUDA: True" not in r.stdout:
    print("FATAL: No CUDA!")
    sys.exit(1)

print("Training BC (20 epochs)...")
r = ssh("cd /workspace && python src/training/train_bc.py --h5 data/processed/XAUUSD_M5_w256.h5 --epochs 20 --window-size 256")
print(r.stdout[-1000:] if r.stdout else r.stderr)

print("Training PPO (500K steps)...")
r = ssh("cd /workspace && python src/training/train_rl.py --h5 data/processed/XAUUSD_M5_w256.h5 --bc-ckpt checkpoints/best_model_bc.pt --n-envs 64 --total-steps 500000")
print(r.stdout[-1000:] if r.stdout else r.stderr)

print("Pulling...")
os.makedirs("checkpoints/fresh_6month", exist_ok=True)
scp_d("/workspace/checkpoints/best_model_bc.pt", "checkpoints/fresh_6month/best_model_bc.pt")
scp_d("/workspace/checkpoints/ppo_best.pt", "checkpoints/fresh_6month/ppo_best.pt")

subprocess.run(["vastai", "destroy", "instance", new_id])
print("Done!")
