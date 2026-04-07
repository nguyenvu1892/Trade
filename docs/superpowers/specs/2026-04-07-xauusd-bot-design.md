# Implementation Design: XAUUSD Deep Reinforcement Learning Bot (Transformer)

## Context & Objective
Mục tiêu là xây dựng một Bot giao dịch AI cho cặp tiền tệ XAUUSD (Vàng/Đô la), lấy cảm hứng từ các hệ thống AI tiên tiến như AlphaZero & AlphaStar. Thay vì code các indicator thủ công, Bot sẽ tự động học các quy luật thị trường từ 10 năm dữ liệu lịch sử thông qua hai giai đoạn:
1. Học bắt chước (Behavioral Cloning - BC)
2. Học tăng cường (Reinforcement Learning - RL)

Để giải quyết bài toán dữ liệu, một "hindsight thuật toán" (Oracle) sẽ được xây dựng để lùi lại quá khứ, vạch ra các điểm vào lệnh (entry) và thoát lệnh (exit) tối ưu nhất, qua đó làm tư liệu gốc cho Bot ở Phase 1. Mô hình "cốt lõi" sẽ là một Transformer Network ưu việt nhằm nắm bắt bối cảnh dài hạn của chuỗi giá.

## System Components

Dự án này sẽ bao gồm 4 khối kiến trúc chính:

### Block 1: Market Environment (Môi trường)
- **Nhiệm vụ:** Chuyển đổi dữ liệu 10 năm nến XAUUSD (OHLCV) nguyên thủy thành Dữ liệu State cho Bot nhìn thấy.
- **Tính năng:**
  - Tích hợp slippage, spread, commision như thị trường thật để tránh việc RL 'overfit' và khai thác lỗ hổng môi trường ảo.
  - Normalization (chuẩn hóa dữ liệu cửa sổ nếp giá).

### Block 2: The Oracle (Máy dò nhãn)
- **Nhiệm vụ:** Scan 10 năm quá khứ với lợi thế "nhìn trước tương lai".
- **Tính năng:**
  - Logic dò tìm những thời điểm giá có Reward/Risk ratio cao (vd: 2:1 trở lên) theo cơ chế Sniper.
  - Quét và trích xuất thành tập dữ liệu `.h5` (HDF5) hoặc CSV gồm `[Window Features, Dấu nhãn Action: Buy/Sell/Hold]`.

### Block 3: The Transformer Brain (Mạng Neural Lõi)
- **Nhiệm vụ:** Làm bộ não cho tác vụ nhận dạng trạng thái thị trường và ra quyết định.
- **Kiến trúc:** 
  - Tokenizer / Embedding cho Price Time-series.
  - Transformer Encoder Layers (áp dụng Self-Attention).
  - Đầu ra là Policy Network (Dự báo xác suất chọn Action) và Value Network (Dự báo Reward tổng thể).

### Block 4: Training Pipeline (Đường ống Huấn luyện)
- **Phase 1 - Behavioral Cloning (BC):** Sử dụng Cross-Entropy Loss để dạy mô hình phân loại (Classification) điểm vào lệnh giống y hệt như những gì Oracle đã làm. 
- **Phase 2 - Reinforcement Learning (PPO Agent):** Sau khi có Win Rate sơ bộ ở Phase 1, đẩy model vào Environment Simulator tự thân thi đấu + phạt/thưởng để tối ưu khả năng Quản lý vốn và chốt lời. Tối ưu theo hàm Sharpe Ratio hoặc Maximum Drawdown Constraint.
- **Support VRAM/Cloud:** Hỗ trợ tính năng checkpointing, distributed training phù hợp để upload lên cụm tính toán RTX 4090/5090 (VD trên Vast.ai).

## Verification & Testing 
- Mọi mô-đun trong Block 1, 2, 3 phải được bao phủ bởi Unit Tests (Test data loading, test Model Forward pass, Test Reward function). Không viết code Model cho đến khi Pipeline Data đã được chạy thử kiểm định.

## Open Questions (For the User)
Tất cả đã rõ ràng, chờ review từ User trước khi chuyển sang kỹ năng sinh kế hoạch thực thi (writing-plans).
