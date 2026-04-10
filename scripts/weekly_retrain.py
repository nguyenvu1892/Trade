"""
weekly_retrain.py — Pipeline Tự Động Retrain Cuối Tuần
=======================================================
1-click: Tải data mới → Build HDF5 → Thuê GPU → Fine-tune → Pull model → Backtest → Deploy

Cách dùng:
  python weekly_retrain.py              # Chạy toàn bộ pipeline
  python weekly_retrain.py --dry-run    # Chỉ tải data + build dataset, không train
"""

import subprocess
import time
import json
import os
import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ═══════════════════════════════════════════════════════════════
# CẤU HÌNH
# ═══════════════════════════════════════════════════════════════

CONFIG = {
    # Data
    "symbol": "XAUUSD",
    "timeframe": "M5",
    "years_back": 1,  # Lấy 1 năm gần nhất (đủ cho fine-tune)
    "window_size": 256,

    # Training (Fine-tune nhẹ)
    "bc_epochs": 20,
    "ppo_steps": 500_000,
    "n_envs": 64,

    # Cloud
    "gpu_filter": "gpu_name=RTX_4090 num_gpus=1 rentable=True verified=true cpu_ram>=64 disk_space>=100 dph_total<=0.5",

    # Model acceptance criteria
    "min_sharpe_ratio": 0.8,    # Model mới phải đạt ≥ 80% Sharpe model cũ
    "max_dd_ratio": 1.1,        # MaxDD model mới ≤ 110% MaxDD model cũ
    "min_win_rate": 0.55,       # Win rate tối thiểu 55%
}

BASH = r"C:\Program Files\Git\bin\bash.exe"
LOG_DIR = Path("logs")
CKPT_DIR = Path("checkpoints")

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


# ═══════════════════════════════════════════════════════════════
# STEP 1: TẢI DATA MỚI TỪ MT5
# ═══════════════════════════════════════════════════════════════

def step1_download_data():
    log("📥 STEP 1: Tải data M5 mới nhất từ MT5...")
    result = subprocess.run(
        [sys.executable, "src/data/download_mt5.py",
         "--symbol", CONFIG["symbol"],
         "--timeframes", CONFIG["timeframe"],
         "--years", str(CONFIG["years_back"])],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        log(f"❌ Tải data thất bại:\n{result.stderr}")
        return False

    log(f"✅ Data đã tải xong")
    print(result.stdout[-500:])
    return True


# ═══════════════════════════════════════════════════════════════
# STEP 2: BUILD DATASET HDF5
# ═══════════════════════════════════════════════════════════════

def step2_build_dataset():
    log("🔧 STEP 2: Build HDF5 dataset...")

    # Tìm file CSV M5 mới nhất
    csv_dir = Path("data/raw")
    csv_files = sorted(csv_dir.glob(f"{CONFIG['symbol']}_{CONFIG['timeframe']}_*.csv"))
    if not csv_files:
        log("❌ Không tìm thấy file CSV M5!")
        return None

    latest_csv = str(csv_files[-1])
    log(f"   CSV: {latest_csv}")

    result = subprocess.run(
        [sys.executable, "src/data/build_dataset.py",
         "--m15", latest_csv,  # build_dataset dùng --m15 flag cho input chung
         ],
        capture_output=True, text=True
    )

    # Tìm file HDF5 output
    h5_dir = Path("data/processed")
    h5_files = sorted(h5_dir.glob(f"*{CONFIG['timeframe']}*w{CONFIG['window_size']}*.h5"))
    if not h5_files:
        # Fallback: tìm bất kỳ h5 nào
        h5_files = sorted(h5_dir.glob("*.h5"), key=os.path.getmtime)

    if h5_files:
        h5_path = str(h5_files[-1])
        log(f"✅ HDF5: {h5_path}")
        return h5_path
    else:
        log(f"❌ Build dataset thất bại:\n{result.stderr}")
        return None


# ═══════════════════════════════════════════════════════════════
# STEP 3: THUÊ GPU VÀ FINE-TUNE
# ═══════════════════════════════════════════════════════════════

def step3_cloud_train(h5_path: str):
    log("☁️ STEP 3: Thuê GPU và Fine-tune...")

    # Tìm GPU
    res = subprocess.run(
        ["vastai", "search", "offers", CONFIG["gpu_filter"], "--on-demand"],
        capture_output=True, text=True
    )
    lines = [l for l in res.stdout.strip().split("\n") if l.strip() and not l.startswith("ID")]
    if not lines:
        log("❌ Không tìm thấy GPU phù hợp!")
        return False

    # Lấy machine ID đầu tiên
    machine_id = lines[0].split()[0]
    log(f"   GPU: Machine #{machine_id}")

    # Thuê instance
    res = subprocess.run(
        ["vastai", "create", "instance", machine_id,
         "--image", "pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime",
         "--disk", "100"],
        capture_output=True, text=True
    )
    if "new_contract" not in res.stdout:
        log(f"❌ Thuê GPU thất bại: {res.stdout}")
        return False

    instance_id = json.loads(res.stdout.split("{", 1)[1].rsplit("}", 1)[0].join(["{", "}"]))["new_contract"]
    instance_id = str(instance_id)
    log(f"   Instance: #{instance_id}")

    # Chờ instance sẵn sàng
    log("   Chờ instance boot...")
    for i in range(60):
        res = subprocess.run(["vastai", "show", "instances", "--raw"], capture_output=True, text=True)
        try:
            data = json.loads(res.stdout)
            inst = next((x for x in data if str(x["id"]) == instance_id), None)
            if inst and inst.get("actual_status") == "running":
                log("   ✅ Instance ready!")
                break
        except:
            pass
        time.sleep(15)
    else:
        log("❌ Instance timeout!")
        subprocess.run(["vastai", "destroy", "instance", instance_id])
        return False

    # Lấy SSH info
    res = subprocess.run(["vastai", "ssh-url", instance_id], capture_output=True, text=True)
    url = res.stdout.strip().replace("ssh://", "")
    parts = url.split(":")
    host = parts[0].split("@")[-1]
    port = parts[1]

    def ssh(cmd):
        return subprocess.run(
            [BASH, "-c", f"ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -p {port} root@{host} '{cmd}'"],
            capture_output=True, text=True
        )

    def scp_up(local, remote):
        subprocess.run(
            [BASH, "-c", f"scp -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -P {port} {local} root@{host}:{remote}"],
            capture_output=True, text=True
        )

    def scp_down(remote, local):
        subprocess.run(
            [BASH, "-c", f"scp -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -P {port} root@{host}:{remote} {local}"],
            capture_output=True, text=True
        )

    # Upload code + data + current model
    log("   📦 Uploading...")
    subprocess.run([BASH, "-c", f"scp -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -P {port} -r src scripts requirements.txt root@{host}:/workspace/"], capture_output=True)
    ssh("mkdir -p /workspace/data/processed /workspace/logs /workspace/checkpoints")
    scp_up(h5_path, "/workspace/data/processed/")
    scp_up("checkpoints/ppo_best.pt", "/workspace/checkpoints/best_model_bc.pt")

    # Install deps + fix CUDA
    log("   🔧 Installing deps...")
    ssh("apt-get update && apt-get install -y dos2unix && cd /workspace && find . -name '*.py' -exec dos2unix {} + && pip install -q -r requirements.txt")
    ssh("pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

    # Fine-tune BC
    h5_name = Path(h5_path).name
    log(f"   🧠 Fine-tune BC ({CONFIG['bc_epochs']} epochs)...")
    ssh(f"cd /workspace && python src/training/train_bc.py --h5 data/processed/{h5_name} --epochs {CONFIG['bc_epochs']} > logs/finetune_bc.log 2>&1")

    # Fine-tune PPO
    log(f"   🧠 Fine-tune PPO ({CONFIG['ppo_steps']//1000}K steps)...")
    ssh(f"cd /workspace && python src/training/train_rl.py --h5 data/processed/{h5_name} --bc-ckpt checkpoints/best_model_bc.pt --n-envs {CONFIG['n_envs']} --total-steps {CONFIG['ppo_steps']} > logs/finetune_rl.log 2>&1")

    # Pull results
    log("   📥 Pulling models...")
    os.makedirs("checkpoints/weekly_candidates", exist_ok=True)
    scp_down("/workspace/checkpoints/ppo_best.pt", "checkpoints/weekly_candidates/ppo_best_new.pt")
    scp_down("/workspace/checkpoints/ppo_final.pt", "checkpoints/weekly_candidates/ppo_final_new.pt")
    scp_down("/workspace/logs/finetune_bc.log", "logs/finetune_bc.log")
    scp_down("/workspace/logs/finetune_rl.log", "logs/finetune_rl.log")

    # Kill server
    log("   💀 Destroying server...")
    subprocess.run(["vastai", "destroy", "instance", instance_id])

    if Path("checkpoints/weekly_candidates/ppo_best_new.pt").exists():
        log("   ✅ Training complete! Model pulled.")
        return True
    else:
        log("   ❌ Model not found after training!")
        return False


# ═══════════════════════════════════════════════════════════════
# STEP 4: BACKTEST & SO SÁNH
# ═══════════════════════════════════════════════════════════════

def step4_compare_models(h5_path: str):
    log("📊 STEP 4: Backtest so sánh model cũ vs mới...")

    import torch
    import h5py
    from src.model.transformer import XAUTransformer
    from src.env.xauusd_env import XAUUSDEnv
    import numpy as np

    with h5py.File(h5_path, "r") as f:
        n_total, window_size, n_features = f["X"].shape
    oos_start = int(n_total * 0.8)

    def backtest_model(ckpt_path, label):
        model = XAUTransformer(n_features=n_features, window_size=window_size,
                               d_model=256, n_heads=8, n_layers=6)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
        else:
            model.load_state_dict(ckpt)
        model.eval()

        env = XAUUSDEnv(h5_path=h5_path, start_idx=oos_start, end_idx=n_total,
                        window_size=window_size, spread_pips=25, lot_size=0.01,
                        initial_balance=200.0, max_drawdown_usd=999999.0, random_start=False)
        obs, _ = env.reset()
        done = False
        equity_hist = [200.0]
        trades_won, trades_lost = 0, 0
        prev_pos = 0
        prev_balance = 200.0

        while not done:
            tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model(tensor)
                probs = torch.softmax(logits, dim=-1).squeeze().numpy()
                action = int(np.argmax(probs))
                conf = float(probs[action])

            # Apply confidence filter
            if action in (1, 2) and conf < 0.45:
                action = 0

            prev_pos = env._position_dir
            prev_balance = env._balance
            obs, _, term, trunc, info = env.step(action)
            equity_hist.append(info.get("equity", equity_hist[-1]))

            if prev_pos != 0 and env._position_dir != prev_pos:
                pnl = env._balance - prev_balance
                if pnl > 0: trades_won += 1
                else: trades_lost += 1

            done = term or trunc

        total_trades = trades_won + trades_lost
        win_rate = trades_won / total_trades if total_trades > 0 else 0

        eq = np.array(equity_hist)
        returns = np.diff(eq) / eq[:-1]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 288)

        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / peak
        max_dd = float(np.max(dd))

        log(f"   {label}: Sharpe={sharpe:.2f}, WR={win_rate:.1%}, MaxDD={max_dd:.1%}, Trades={total_trades}, Equity=${eq[-1]:.2f}")
        return {"sharpe": sharpe, "win_rate": win_rate, "max_dd": max_dd, "final_eq": eq[-1]}

    old = backtest_model("checkpoints/ppo_best.pt", "CURRENT MODEL")
    new = backtest_model("checkpoints/weekly_candidates/ppo_best_new.pt", "NEW MODEL    ")

    return old, new


def step5_deploy_or_reject(old_metrics, new_metrics):
    log("🎯 STEP 5: Quyết định deploy...")

    sharpe_ok = new_metrics["sharpe"] >= old_metrics["sharpe"] * CONFIG["min_sharpe_ratio"]
    dd_ok = new_metrics["max_dd"] <= old_metrics["max_dd"] * CONFIG["max_dd_ratio"]
    wr_ok = new_metrics["win_rate"] >= CONFIG["min_win_rate"]

    log(f"   Sharpe: {new_metrics['sharpe']:.2f} vs {old_metrics['sharpe']:.2f} × {CONFIG['min_sharpe_ratio']} = "
        f"{old_metrics['sharpe']*CONFIG['min_sharpe_ratio']:.2f} → {'✅' if sharpe_ok else '❌'}")
    log(f"   MaxDD:  {new_metrics['max_dd']:.1%} vs {old_metrics['max_dd']:.1%} × {CONFIG['max_dd_ratio']} = "
        f"{old_metrics['max_dd']*CONFIG['max_dd_ratio']:.1%} → {'✅' if dd_ok else '❌'}")
    log(f"   WR:     {new_metrics['win_rate']:.1%} vs {CONFIG['min_win_rate']:.0%} → {'✅' if wr_ok else '❌'}")

    if sharpe_ok and dd_ok and wr_ok:
        # Deploy!
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        backup_name = f"checkpoints/ppo_best_backup_{timestamp}.pt"
        shutil.copy2("checkpoints/ppo_best.pt", backup_name)
        shutil.copy2("checkpoints/weekly_candidates/ppo_best_new.pt", "checkpoints/ppo_best.pt")
        log(f"🏆 MODEL DEPLOYED! Backup → {backup_name}")
        log(f"   Restart live_bot.py để dùng model mới.")
        return True
    else:
        log("⚠️  Model mới KHÔNG đạt tiêu chuẩn. Giữ model cũ.")
        return False


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Chỉ tải data + build, không train")
    parser.add_argument("--skip-download", action="store_true", help="Bỏ qua bước tải data")
    args = parser.parse_args()

    print("=" * 60)
    print("🔄 SCALPEX200 — WEEKLY RETRAIN PIPELINE")
    print(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"   BC Epochs: {CONFIG['bc_epochs']} | PPO Steps: {CONFIG['ppo_steps']//1000}K")
    print("=" * 60)

    # Step 1: Download
    if not args.skip_download:
        if not step1_download_data():
            log("Pipeline dừng tại Step 1.")
            return
    else:
        log("⏭️ Bỏ qua download data")

    # Step 2: Build dataset
    h5_path = step2_build_dataset()
    if h5_path is None:
        log("Pipeline dừng tại Step 2.")
        return

    if args.dry_run:
        log("🏁 Dry-run hoàn tất. Dùng --skip-download để chạy train tiếp.")
        return

    # Step 3: Cloud train
    if not step3_cloud_train(h5_path):
        log("Pipeline dừng tại Step 3.")
        return

    # Step 4: Compare
    old, new = step4_compare_models(h5_path)

    # Step 5: Deploy or reject
    deployed = step5_deploy_or_reject(old, new)

    # Summary
    print("\n" + "=" * 60)
    if deployed:
        print("🏆 WEEKLY RETRAIN THÀNH CÔNG!")
        print("   Model mới đã deploy. Restart live_bot.py.")
    else:
        print("📋 WEEKLY RETRAIN HOÀN TẤT")
        print("   Model cũ vẫn được giữ (model mới không đạt chuẩn).")
    print("=" * 60)


if __name__ == "__main__":
    main()
