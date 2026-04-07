# Design Spec: XAUUSD Deep RL Trading Bot (AlphaStar-Inspired)

**Version:** 2.0 — Post User Review  
**Date:** 2026-04-07  
**Sàn:** Exness Raw Spread | **Vốn:** $200 | **Lot:** 0.01 cố định

---

## Context & Objective

Xây dựng một Bot giao dịch AI cho cặp XAUUSD lấy cảm hứng từ AlphaZero & AlphaStar. Bot học hoàn toàn từ dữ liệu, không dùng indicator cứng nhắc. Quá trình huấn luyện 2 pha:

1. **Phase 1 — Behavioral Cloning (BC):** Học bắt chước một thuật toán Oracle "nhìn trước tương lai".
2. **Phase 2 — Reinforcement Learning (PPO):** Tự tối ưu chiến lược quản trị rủi ro thông qua tương tác với môi trường mô phỏng.

**Hardware:** Thuê GPU server RTX 4090 / RTX 5090 (Vast.ai). Training pipeline phải hỗ trợ Docker, mixed-precision fp16/bf16, Vectorized Environments.

---

## Tech Stack (Đã chốt)

| Thành phần | Thư viện |
|---|---|
| Lõi tính toán | PyTorch |
| Môi trường Gym | Gymnasium |
| PPO Agent | Stable-Baselines3 hoặc Ray RLlib |
| I/O Dữ liệu lớn | H5py (HDF5) |
| Containerization | Docker |

---

## System Architecture — 4 Khối chính

---

### Block 1: Data Pipeline & Feature Engineering

**Nhiệm vụ:** Chuyển đổi dữ liệu OHLCV thô thành Feature Tensor chuẩn hóa, khử phi dừng.

**Timeframe data đầu vào:** M5, M15, H1 (multi-timeframe — 3 khung cùng được nạp vào pipeline).

#### 1.1 — Khử Tính Phi Dừng (Non-Stationarity)

Giá vàng tuyệt đối (Absolute Price) không thể đưa thẳng vào Transformer. Giá năm 2015 quanh $1,100 — hiện tại $2,300+. Mô hình sẽ bị **Out-of-Distribution (OOD)** hoàn toàn với các mức giá tương lai chưa gặp khi train.

**Giải pháp bắt buộc — Log Returns:**
```
log_return(t) = ln( Close(t) / Close(t-1) )
```
- Biến đổi giá tuyệt đối thành phần trăm thay đổi tương đối.
- Log returns có phân phối ổn định theo thời gian (stationary) — mô hình học được quy luật chung, không bị gắn vào một vùng giá cụ thể.
- Áp dụng cho cả O, H, L, C (relative to previous close) và Volume.

**Bổ sung — Fractional Differentiation (tuỳ chọn nâng cao):**
Kỹ thuật của Marcos Lopez de Prado cho phép lấy vi phân một lượng nhỏ (fraction `d < 1`) thay vì vi phân nguyên, giữ lại nhiều "bộ nhớ" của chuỗi giá hơn so với log returns thuần túy. Sẽ đánh giá sau Sprint 1 nếu log returns không đủ.

#### 1.2 — Nhúng Ngữ Cảnh Thời Gian (Temporal Context)

Vàng có tính thời vụ rõ rệt: phiên Á đi ngang, phiên Âu/Mỹ biến động mạnh. Mô hình phải biết "bây giờ là mấy giờ".

**Sine/Cosine Encoding bắt buộc cho:**
```python
# Hour of day (0-23) → tuần hoàn 24h
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)

# Day of week (0-6) → tuần hoàn 7 ngày
dow_sin  = sin(2π × day_of_week / 7)
dow_cos  = cos(2π × day_of_week / 7)
```
Encoding dạng này đảm bảo 23:00 và 00:00 "gần nhau" trong không gian embedding, không bị đứt gãy.

#### 1.3 — Feature Vector cuối cùng mỗi timestep

```
[log_ret_O, log_ret_H, log_ret_L, log_ret_C, log_vol,
 ATR_norm, RSI_norm, MACD_norm,
 hour_sin, hour_cos, dow_sin, dow_cos]
```

---

### Block 2: The Oracle — Triple Barrier Method

**Nhiệm vụ:** Tự động gắn nhãn [Hold=0, Buy=1, Sell=2] cho 10 năm dữ liệu một cách trung thực — có tính đến rủi ro drawdown giữa điểm vào và điểm thoát.

#### Tại sao KHÔNG dùng Oracle đơn giản (điểm B > điểm A)?

Oracle kiểu "giá cuối cao hơn giá đầu = Buy" bỏ qua hoàn toàn việc **giá có thể nhúng sâu xuống điểm C ở giữa** — đủ để quét sạch Stop Loss thực tế trước khi hồi. Nhãn Buy trong trường hợp này là **nhãn giả** — không thể trade được trên thực tế.

#### Giải pháp: Triple Barrier Method (Lopez de Prado)

Tại mỗi điểm thời gian `t`, dựng 3 rào cản:

```
Barrier 1 (TP):  Giá tăng +X% từ entry → Label = BUY (Win)
Barrier 2 (SL):  Giá giảm -Y% từ entry → Label = HOLD (Filtered out / invalid)
Barrier 3 (Time): Hết T nến mà chưa chạm TP hay SL → Label = HOLD (Expires)
```

**Oracle chỉ gắn nhãn BUY/SELL hợp lệ khi giá chạm TP TRƯỚC khi chạm SL.**  
Tương tự cho hướng Sell.

**Tham số ban đầu (dựa trên $200 vốn, 0.01 lot):**
- TP: +1.5 ATR
- SL: -1.0 ATR  
- Max hold: 48 nến (12 giờ với M15)

Kết quả: Dataset gồm các lệnh "thực sự có thể thắng" — không có nhãn ảo.

#### Xử lý Class Imbalance (Mất cân bằng nhãn)

Trong 10 năm, nhãn Hold chiếm **~90–95%** tổng số điểm dữ liệu. Đây là bẫy chết người:

> **Vấn đề:** Model đạt accuracy 92% chỉ bằng cách đoán "Hold" cho MỌI điểm — nhưng P&L = 0.

**Giải pháp bắt buộc trong Phase 1 (BC Training):**

1. **Focal Loss** (thay Cross-Entropy thông thường):
   ```
   FL(pt) = −α_t × (1 − pt)^γ × log(pt)
   ```
   - `γ` (focusing parameter, mặc định = 2): phạt nặng hơn khi model predict sai nhãn hiếm (Buy/Sell).
   - `α_t`: trọng số ngược tần suất — nhãn hiếm hơn được bảo vệ hơn.

2. **Class Weights** bổ trợ: tính tự động từ tần suất thực tế của dataset.

**Metric đánh giá:** KHÔNG dùng Accuracy. Dùng **F1-Score riêng cho từng nhãn Buy & Sell** trên tập validation. Target: F1(Buy) > 0.4, F1(Sell) > 0.4 trước khi chuyển Phase 2.

---

### Block 3: The Transformer Brain

**Nhiệm vụ:** Bộ não ra quyết định dựa trên chuỗi Feature Tensor.

#### Kiến trúc Causal Transformer Encoder

Sử dụng **Causal Mask** (triangular attention mask) — nến tại timestep `t` **chỉ được phép nhìn thấy các nến từ `t-N` đến `t`**, không nhìn tương lai. Đây là yêu cầu bắt buộc để tránh data leakage ngay trong quá trình training.

```
Input Sequence: [t-N, t-N+1, ..., t]  (N = window_size, dự kiến 128 nến)
     ↓
Price Embedding Layer (Linear projection → d_model)
     ↓
Positional Encoding (Sine/Cosine absolute positions)
     ↓
Causal Transformer Encoder × L layers
(Multi-Head Self-Attention + FFN + LayerNorm + Dropout)
     ↓
Global Average Pooling (lấy đại diện toàn chuỗi)
     ↓
    ┌─────────────────────┐
    ↓                     ↓
Policy Head          Value Head
(Linear → 3 logits)  (Linear → 1 scalar)
[Hold, Buy, Sell]    [Expected Return]
```

**Tham số dự kiến:** d_model=256, heads=8, layers=6, dropout=0.1.

---

### Block 4: Training Pipeline

#### Chống Rò Rỉ Dữ Liệu (Data Leakage) — Purged Walk-Forward Validation

**KHÔNG ĐƯỢC dùng Random Split hay K-Fold** cho time-series. Các điểm dữ liệu liền kề nhau tương quan cao — random split sẽ "rò" thông tin tương lai vào tập train.

**Bắt buộc dùng Purged Walk-Forward:**
```
[=====TRAIN=====][GAP][==VAL==][GAP][==TEST (Out-of-sample)==]
     8 năm        1th   1 năm   1th      1 năm cuối (2024+)
```
- **GAP (Purge period):** Xóa bỏ một khoảng thời gian giữa Train và Val/Test để loại bỏ hoàn toàn hiệu ứng tương quan chuỗi thời gian.
- **Test set hoàn toàn bị khóa**, không được "nhìn" trong suốt quá trình phát triển.

---

#### Phase 1 — Behavioral Cloning (BC)

- **Loss:** Focal Loss với class weights.
- **Optimizer:** AdamW với cosine LR scheduler.
- **Early stopping:** Dựa trên F1-Score(Buy+Sell) trên validation set.
- **Điều kiện chuyển Phase 2:** F1(Buy) > 0.4 AND F1(Sell) > 0.4 AND Win Rate validate > 55%.

---

#### Phase 2 — PPO Reinforcement Learning

**Chống Catastrophic Forgetting (Quên thảm khốc):**

Khi PPO bắt đầu khám phá ngẫu nhiên, nó dễ phá hủy các weights quý báu đã học từ Oracle ở Phase 1. Giải pháp: **KL-Divergence Anchor** trong hàm Loss PPO:

```
Total PPO Loss = PPO_Clip_Loss + λ(t) × KL(π_PPO || π_BC)
```
- `KL(π_PPO || π_BC)`: đo mức độ Policy PPO "xa rời" Policy BC đã train.
- `λ(t)`: giảm dần từ 0.5 → 0.05 theo số training steps — cho phép Bot vượt trội Oracle dần dần.

**Reward Function hoàn chỉnh (có tính $200 vốn, 0.01 lot):**

```python
R(t) = (
    pnl_on_close(t)           # USD thực khi đóng lệnh (~$0.10 per $1 move)
  - opportunity_cost(t)       # = missed_profit nếu Oracle=Trade AND Bot=Hold
  - holding_cost(t)           # = 0.001 × consecutive_hold_bars (cộng dồn)
  + sharpe_bonus(episode)     # thưởng cuối episode nếu Sharpe > 1.0
  - drawdown_penalty(t)       # phạt nặng nếu equity drawdown > $20 (10% account)
)
# Episode kết thúc sớm nếu equity < $180 (drawdown $20)
```

**Vectorized Environments:** Chạy 64–128 bản sao môi trường song song trên GPU server để tăng tốc thu thập kinh nghiệm.

---

## Verification & Testing Strategy

Mọi module **phải có passing tests trước khi viết code tiếp theo** (TDD).

| Module | Test Cases |
|---|---|
| Data Processor | Log returns không chứa NaN/Inf; shape output đúng; Sine/Cosine nằm trong [-1,1] |
| Oracle (Triple Barrier) | Feed giá giả lập có đỉnh/đáy rõ ràng → assert nhãn đúng; assert nhãn chạm SL bị lọc bỏ |
| Market Environment | `step()` tính Balance/Equity/Margin khớp MT4; episode dừng đúng khi chạm drawdown $20 |
| Transformer Forward Pass | Dummy tensor → output shape [batch, 3] (Policy) và [batch, 1] (Value); Gradient lưu thông tới lớp sâu nhất |
| BC Training Loop | Loss giảm qua epochs; F1(Buy) và F1(Sell) trên val không bằng 0 |

---

## Execution Plan — 4 Sprints (TDD)

### Sprint 1: Data Foundation & Oracle
**Mục tiêu:** Dataset "hoàn hảo" cho Phase 1.

- **Task 1.1:** `data_processor.py` — nạp M15/H1 CSV, tính Log Returns, ATR, Sine/Cosine time encoding.
- **Task 1.2:** `oracle.py` — Triple Barrier Method, output nhãn [0,1,2].
- **Task 1.3:** `dataset_builder.py` — sliding window, đóng gói ra file `.h5`.
- **Tests:** `test_data_processor.py`, `test_oracle.py`.

### Sprint 2: Market Simulator (Gymnasium Env)
**Mục tiêu:** Môi trường cọ xát chính xác như Exness thật.

- **Task 2.1:** `xauusd_env.py` kế thừa `gymnasium.Env`.
- **Task 2.2:** Tích hợp Spread biến động theo giờ, Commission, Slippage, Margin call.
- **Tests:** `test_env.py` — check PnL formula, check episode termination, memory leak test (1M steps).

### Sprint 3: Transformer Brain & BC Training (Phase 1)
**Mục tiêu:** Win Rate validate > 55%, F1(Buy+Sell) > 0.4.

- **Task 3.1:** `model.py` — Causal Transformer Encoder, Policy Head, Value Head.
- **Task 3.2:** `train_bc.py` — Focal Loss, Walk-Forward split, training loop, checkpoint.
- **Tests:** `test_model.py` — forward pass, gradient flow, output shape.

### Sprint 4: RL Fine-tuning & Cloud Scale-up (Phase 2)
**Mục tiêu:** Tối ưu Drawdown, Sharpe trên out-of-sample.

- **Task 4.1:** Wrap pre-trained model vào PPO Policy (SB3 hoặc RLlib).
- **Task 4.2:** Reward function hoàn chỉnh với KL-Divergence anchor.
- **Task 4.3:** `Dockerfile` + launch script cho Vast.ai (Vectorized Envs, 64 workers).
- **Task 4.4:** Out-of-sample backtest — report Sharpe, Sortino, Max Drawdown, Win Rate.

---

## Capital & Execution Constraints (Đã chốt)

| Tham số | Giá trị |
|---|---|
| Sàn | Exness — Raw Spread |
| Vốn khởi điểm | $200 USD |
| Lot size | 0.01 (cố định, không martingale) |
| P&L mỗi $1 giá vàng | ~$0.10 |
| Max Drawdown / Episode | $20 (10%) — episode kết thúc sớm + penalty nặng |
| Mục tiêu live v1 | Win Rate > 55%, Sharpe > 1.0, Max DD < 10% |

---

## Data Source (Đã chốt)

- **Nguồn:** MetaTrader5 Python API kết nối trực tiếp với Exness MT5 Terminal.
- **Timeframe:** M5, M15 và H1, kéo về 10 năm lịch sử.
- **Yêu cầu:** MT5 Exness terminal phải đang chạy và đã đăng nhập trên máy Windows.
- **Script:** `src/data/download_mt5.py`
