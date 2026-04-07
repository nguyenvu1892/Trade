# XAUUSD AI Trading Bot

Bot giao dịch AI cho cặp XAUUSD, lấy cảm hứng từ AlphaZero & AlphaStar.

## Kiến trúc
- **Phase 1:** Behavioral Cloning từ Oracle (Triple Barrier Method)
- **Phase 2:** PPO Reinforcement Learning với KL-Divergence anchor

## Cấu trúc thư mục
```
Trade/
├── data/
│   └── raw/              # CSV data từ MT5 (M15, H1)
├── docs/
│   └── superpowers/specs/
│       └── 2026-04-07-xauusd-bot-design.md
├── src/
│   ├── data/
│   │   ├── download_mt5.py       # Sprint 1: Tải data từ Exness MT5
│   │   ├── data_processor.py     # Sprint 1: Log returns, Sine/Cosine encoding
│   │   ├── oracle.py             # Sprint 1: Triple Barrier Method labeler
│   │   ├── dataset_builder.py    # Sprint 1: Đóng gói HDF5 dataset
│   │   └── tests/
│   ├── env/
│   │   ├── xauusd_env.py         # Sprint 2: Gymnasium environment
│   │   └── tests/
│   ├── model/
│   │   ├── transformer.py        # Sprint 3: Causal Transformer Brain
│   │   └── tests/
│   └── training/
│       ├── train_bc.py           # Sprint 3: Behavioral Cloning loop
│       ├── train_rl.py           # Sprint 4: PPO fine-tuning
│       └── backtest.py           # Sprint 4: Out-of-sample evaluation
├── requirements.txt
└── README.md
```

## Cách bắt đầu

### 1. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 2. Tải dữ liệu (cần MT5 Exness đang chạy)
```bash
python src/data/download_mt5.py
```
Hoặc chỉ định tham số:
```bash
python src/data/download_mt5.py --symbol XAUUSD --timeframes M15 H1 --years 10
```

### 3. Chạy tests
```bash
pytest src/ -v --cov=src
```

## Thông số kỹ thuật
- **Sàn:** Exness Raw Spread
- **Symbol:** XAUUSD
- **Vốn:** $200 | **Lot:** 0.01 (cố định)
- **Max Drawdown:** $20 (10%)
- **Hardware training:** RTX 4090 / RTX 5090 (Vast.ai)
