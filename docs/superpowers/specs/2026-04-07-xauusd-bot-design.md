# Implementation Design: XAUUSD Deep Reinforcement Learning Bot (Transformer)

## Context & Objective
Mục tiêu là xây dựng một Bot giao dịch AI cho cặp tiền tệ XAUUSD (Vàng/Đô la), lấy cảm hứng từ các hệ thống AI tiên tiến như AlphaZero & AlphaStar. Thay vì code các indicator thủ công, Bot sẽ tự động học các quy luật thị trường từ 10 năm dữ liệu lịch sử thông qua hai giai đoạn:
1. Học bắt chước (Behavioral Cloning - BC)
2. Học tăng cường (Reinforcement Learning - RL)

Để giải quyết bài toán dữ liệu, một "hindsight thuật toán" (Oracle) sẽ được xây dựng để lùi lại quá khứ, vạch ra các điểm vào lệnh (entry) và thoát lệnh (exit) tối ưu nhất, qua đó làm tư liệu gốc cho Bot ở Phase 1. Mô hình "cốt lõi" sẽ là một Transformer Network ưu việt nhằm nắm bắt bối cảnh dài hạn của chuỗi giá.

Phần cứng huấn luyện: Thuê GPU server RTX 4090 / RTX 5090 (ví dụ Vast.ai) để xử lý khối lượng tính toán lớn trong thời gian hợp lý.

---

## System Components

Dự án này sẽ bao gồm 4 khối kiến trúc chính:

### Block 1: Market Environment (Môi trường)
- **Nhiệm vụ:** Chuyển đổi dữ liệu 10 năm nến XAUUSD (OHLCV) nguyên thủy thành Dữ liệu State cho Bot nhìn thấy.
- **Tính năng:**
  - Tích hợp slippage, spread, commission như thị trường thật để tránh việc RL 'overfit' và khai thác lỗ hổng môi trường ảo.
  - Normalization (chuẩn hóa dữ liệu theo cửa sổ nến động - rolling window).
  - Cung cấp đầy đủ API chuẩn Gym: `reset()`, `step(action)`, `render()`.

### Block 2: The Oracle (Máy dò nhãn chuyên gia)
- **Nhiệm vụ:** Scan 10 năm quá khứ với lợi thế "nhìn trước tương lai" (hindsight) để tự động tạo ra dataset huấn luyện chất lượng cao.
- **Tính năng:**
  - Logic dò tìm những thời điểm giá có Reward/Risk ratio cao (≥ 2:1 trở lên) theo cơ chế Sniper.
  - Quét và trích xuất thành tập dữ liệu `.h5` (HDF5) gồm `[Window Features (OHLCV + Indicators), Label Action: Buy/Sell/Hold]`.
  - Output: Hàng trăm nghìn điểm vào lệnh hoàn hảo mà không trader nào có thể làm được bằng tay.

### Block 3: The Transformer Brain (Mạng Neural Lõi)
- **Nhiệm vụ:** Làm bộ não cho tác vụ nhận dạng trạng thái thị trường và ra quyết định.
- **Kiến trúc:**
  - Price Embedding Layer: chiếu OHLCV + indicators thành vector embedding.
  - Positional Encoding: mã hóa vị trí thời gian của từng nến.
  - Transformer Encoder Layers (Multi-Head Self-Attention): phân tích mức độ liên quan giữa nến hiện tại với toàn bộ lịch sử trong cửa sổ.
  - Đầu ra tách thành hai nhánh:
    - **Policy Head**: output xác suất cho 3 hành động [Buy, Sell, Hold].
    - **Value Head**: output ước lượng tổng phần thưởng kỳ vọng từ trạng thái hiện tại.

### Block 4: Training Pipeline (Đường ống Huấn luyện)

#### Phase 1 — Behavioral Cloning (BC)
- Dùng Cross-Entropy Loss để huấn luyện Policy Head bắt chước chính xác nhãn của Oracle.
- Kết thúc khi đạt Win Rate trên tập validate ≥ 55% (ngưỡng tối thiểu đủ "cảm giác thị trường").

#### Phase 2 — Reinforcement Learning (PPO + BC Anchor)
Sau khi có nền tảng BC, đẩy model vào Environment Simulator tự thân khám phá.

**Giải pháp chống Reward Hacking / Inaction Bias:**

Vấn đề: Nếu chỉ dùng hàm thưởng/phạt đơn thuần, Bot sẽ học cách "lách luật" bằng cách đứng im (Hold mãi mãi) vì "không làm gì = không bị phạt". Giải pháp được thiết kế theo 3 lớp bảo vệ, lấy cảm hứng trực tiếp từ cách DeepMind giải quyết vấn đề này trong AlphaStar:

**Lớp 1 — Opportunity Cost Penalty (Phạt Chi Phí Cơ Hội):**
Khi Oracle xác định đây là thời điểm tốt để vào lệnh mà Bot chọn Hold, Bot bị phạt một khoản tương đương lợi nhuận bị bỏ lỡ:
```
r_opportunity(t) = −missed_profit  nếu Oracle=Trade AND Bot=Hold
```
Phá vỡ logic "Hold = An toàn". Hold sai thời điểm tệ ngang với Trade thua.

**Lớp 2 — BC Anchor Loss / KL Divergence Regularizer (Mỏ neo hành vi — bí quyết AlphaStar):**
Thay vì tắt BC sau Phase 1, giữ nguyên BC Loss như một hạng tử điều chuẩn trong suốt quá trình RL:
```
Total Loss = RL_Loss + λ(t) × KL( π_RL || π_Oracle )
```
- `KL Divergence` đo mức độ "xa rời" của Bot so với Oracle.
- Nếu Bot drift quá xa (ví dụ: toàn Hold), nó bị kéo ngược lại gần hành vi Oracle.
- `λ(t)` giảm dần theo epoch để Bot vẫn có không gian khám phá và vượt trội Oracle.

**Lớp 3 — Temporal Holding Cost (Phí Cơ Hội Đứng Im):**
Mỗi timestep Bot đứng yên không có vị thế nhận một khoản phạt nhỏ cộng dồn:
```
r_holding(t) = −0.001 × consecutive_hold_bars
```
Buộc Bot phải liên tục ra quyết định thay vì tê liệt vĩnh cửu.

**Hàm Reward tổng thể:**
```
R(t) = PnL(t)                     # +/- tùy thắng/thua sau khi đóng lệnh
     − opportunity_cost(t)        # phạt bỏ lỡ cơ hội Oracle
     − holding_cost(t)            # phạt mỗi nến đứng yên
     + sharpe_bonus(episode)      # thưởng cuối episode nếu Sharpe Ratio > 1.0
     − drawdown_penalty(t)        # phạt nếu drawdown vượt ngưỡng cho phép (vd: 15%)
```

**Support VRAM/Cloud:** Hỗ trợ gradient checkpointing, mixed-precision (fp16/bf16), và distributed training để tối ưu cho GPU server thuê.

---

## Verification & Testing
- Mọi module (Block 1, 2, 3) phải được bao phủ bởi Unit Tests trước khi viết code tiếp theo:
  - Test data loading & normalization (Block 1).
  - Test Oracle label generation & distribution (Block 2): kiểm tra tỉ lệ Buy/Sell/Hold không bị lệch quá mức.
  - Test Model forward pass (Block 3): input → output shape đúng, không NaN.
  - Test Reward function (Block 4): verify từng thành phần reward tính đúng.
- Không viết code Model cho đến khi Pipeline Data đã được chạy và validate xong.

---

## Constraints & Decisions Made
- **Language:** Python 3.10+
- **Framework:** PyTorch (primary), Stable-Baselines3 hoặc custom PPO cho RL phase.
- **Data format:** HDF5 (`.h5`) cho dataset huấn luyện, CSV cho dữ liệu giá thô đầu vào.
- **Timeframe data đầu vào:** Cần xác định trong bước tiếp theo (M1, M5, H1, D1 hoặc multi-timeframe).
- **Actions:** 3 action rời rạc: `[0=Hold, 1=Buy, 2=Sell]`.

---

## Open Questions (Cần xác nhận trước khi viết plan)
1. **Timeframe nến:** Bot sẽ đọc dữ liệu ở khung thời gian nào? (H1, M15, M5, hoặc nhiều khung kết hợp?)
2. **Nguồn dữ liệu:** Bạn có sẵn file CSV lịch sử XAUUSD chưa, hay chúng ta cần viết script tải từ API (ví dụ: MetaTrader export, yfinance, Alpha Vantage)?
