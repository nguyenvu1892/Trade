# XAUUSD AI Trading Bot Ã¢â‚¬â€ Training Container (Phase 2: PPO)
# TÃ¡Â»â€˜i Ã†Â°u cho RTX 4090 / RTX 5090 trÃƒÂªn Vast.ai
# [FIX] Phase 2 (PPO) dÃƒÂ¹ng AsyncVectorEnv Ã¢â‚¬â€ cÃ¡ÂºÂ§n CPU Ã„â€˜a luÃ¡Â»â€œng mÃ¡ÂºÂ¡nh,
# khÃƒÂ´ng chÃ¡ÂºÂ¡y BC (Phase 1) vÃƒÂ¬ BC Ã„â€˜ÃƒÂ£ xong rÃ¡Â»â€œi!

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    htop \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements vÃƒÂ  install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY data/ ./data/

# TÃ¡ÂºÂ¡o thÃ†Â° mÃ¡Â»Â¥c output
RUN mkdir -p checkpoints logs

# [FIX] CMD chÃ¡ÂºÂ¡y PPO (Phase 2), khÃƒÂ´ng phÃ¡ÂºÂ£i BC (Phase 1)
# BC checkpoint truyÃ¡Â»Ân vÃƒÂ o qua --bc-ckpt khi docker run
CMD ["python", "src/training/train_rl.py", \\
     "--h5",          "data/processed/XAUUSD_M15_w128.h5", \\
     "--bc-ckpt",     "checkpoints/best_model_bc.pt", \\
     "--n-envs",      "64", \\
     "--total-steps", "2000000"]