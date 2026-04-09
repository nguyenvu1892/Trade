---
description: Quy trình Deploy NOHUP CRASH-PROOF trên Cloud / Vast.AI
---

# Cloud Deployment NOHUP / CRASH-PROOF Workflow

Đây là Quy trình chuẩn **BẮT BUỘC SỬ DỤNG** mỗi khi đưa kịch bản lên các Node ở Vast.AI hay bất kỳ Cloud Provider nào khác, để phá tan rủi ro đứt mạng SSH, phình dung lượng băng thông, và mất mát dữ liệu thực tế.

## Bước 1: Khởi Tạo Instance & Validate Requirements (Luật Thuê Datacenter Tối Thượng)
BẮT BUỘC áp dụng cờ Search cứng theo 5 nguyên tắc sau để thuê GPU:
1. **GPU Optimal:** Ưu tiên số một cho `RTX_5090` (Burst Compute Mode: Tăng tốc độ >50% giúp tiết kiệm Tổng Phí và Thời Gian chạy).
2. **Loại máy (Hosting Type):** `datacenter=True verified=true`. Phải đặt cọc theo kiểu `On-Demand=True`. Không thuê máy P2P yếu rớt mạng.
3. **RAM Tiêu chuẩn:** Tối thiểu `64GB RAM` đai thực (`cpu_ram>=64`) chống OOM.
4. **Ổ cứng (Disk Space):** BẮT BUỘC trống từ `100GB`.
5. **Chi Phí Tối Ưu:** Bắt đỉnh giá mới `dph <= 0.8`. Chi phí cực hạn này vẫn vô cùng tiết kiệm vì tốc độ xử lý trả kết quả siêu nhanh.

// turbo
**Lệnh Tìm Kiếm Bắt Buộc:**
vastai search offers "gpu_name=RTX_5090 rentable=True datacenter=True verified=true cpu_ram>=64 disk_space>=100 dph<=0.8" --on-demand --raw

## Bước 2: Giao Thức Nhảy Cóc Băng Thông (SCP Upload Protocol)
Tuyệt đối KHÔNG Bơm lên Cloud dữ liệu rác Raw (CSV, Parquet M1 khổng lồ hay SQL Blobs). Truyền dữ liệu kiểu này sẽ thiêu rụi chi phí lưu trữ và tắc nghẽn Upload Network.
- **Lệnh Tối Ưu Mới:** "Chém bỏ Data Cục SQL. Mình chỉ Bơm lên Vast File Source Cùng Chính Xác Thư Mục `data/processed/*.hdf5`". Việc truyền tải Cực Khủng Mà Siêu Siêu Nhẹ Tênh Tải Chỉ Mất Mấy Phút!
- Sử dụng lệnh `scp` đẩy chính xác file `.hdf5` Tinh Luyện vào `workspace` của Cloud Cùng với toàn bộ mã nguồn.

## Bước 3: Diệt Trừ Mã Độc CRLF Của Windows
Do hệ điều hành Local là Windows \r\n, Bash trên Cloud sẽ chết cục bộ. Chuẩn HOÁ LIỀN LẬP TỨC trên máy ảo:

// turbo
find . -type f \( -name '*.sh' -o -name '*.py' \) -exec dos2unix {} +

## Bước 4: Chạy Máy Trực Chiến Bằng Đai Sinh Tồn `tmux` / `nohup`
Tuyệt đối không chạy lệnh Python Train thẳng qua Session SSH Mở Mặc Định (Stdout) vì Gateway Firewall của Vast.AI sẽ đá văng (Timeout Kick) và dừng script giữa chừng.
- **Đai Sinh Tồn `tmux` (Terminal Multiplexer):** Lệnh chạy Training ở Vast BẮT BUỘC được cấy qua phân khu của `tmux` kết hợp `NOHUP`. Mạng Internet ở nhà Cục Bộ mất, Cáp Rớt, hoặc Cụt Nguồn Laptop đi ngủ? Kệ Nó, Cọc GPU trên Mây Vẫn Đều Đều Training Xuyên Buổi Tối Tảo Báo Epoch Thành Công Trả Lại Proxy Cởi Nút Lãi Cho Model!
  - Lệnh tạo: `tmux new -s train_session`
  - Lệnh phóng lửa: `nohup python scripts/train.py > cloud_train.log 2>&1 &`
  - Thoát đai bảo vệ: Bấm `Ctrl+B` rồi bấm `D` (Detach).

## Bước 5: Kiểm Chứng & Diagnostic Đọc Log Nguội
Tuyệt đối không đoán mò nguyên nhân bug. Nếu Fail, phải mở file `cloud_train.log` kiểm tra toàn bộ Traceback đỏ. Tôn trọng tuyệt đối Systematic Debugging.

## Bước 6: Lấy Nhanh Checkpoint và Clean Up
1. Không Đợi Model Cày Đủ Epoch. Nếu Có Model Mới Phá Mốc Kỷ Lục `Proxy PnL` Tạo Ra Lãi. Dùng lệnh `scp` (hoặc rsync) Kéo File Nhỏ `best_model.pt` Trở Lại Ngay Máy Local để test Numba Backtest Simulator.
2. **Quyết Định Cleanup Bắt Buộc (Trảm Quyết):**
   - **TIÊU DIỆT TÀN DƯ:** Tuyệt đối không giữ lại hay ngủ đông Server sau khi Train xong để lãng phí dù chỉ 1 Xu phí Disk. Ngay khi rút được Model Lãi thành công, BẮT BUỘC vạch ngay LỆNH `vastai destroy instance <ID>` KIÊN QUYẾT XÓA BỎ VĨNH VIỄN Instance để cắt đứt mọi vòi Phí Thuê Máy!
