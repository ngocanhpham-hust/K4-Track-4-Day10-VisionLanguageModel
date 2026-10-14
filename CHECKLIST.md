# Checklist tiến độ Lab Day 10

Đánh dấu `[x]` ngay sau khi có bằng chứng thật. Không đánh dấu các mục GPU chỉ
dựa trên code đã viết.

## Repository và môi trường

- [x] `origin` trỏ tới fork GitHub cá nhân.
- [x] Repository GitHub đang ở chế độ Public.
- [x] Không ghi Kaggle token trong source/notebook.
- [x] Bộ test công cụ đề bài đạt `68 passed, 1 skipped` trước khi chạy dữ liệu.
- [x] Kaggle Settings đã bật GPU và Internet.
- [x] `nvidia-smi -L` nhận T4 và working directory nằm dưới `/kaggle/working`.
- [x] Đã tải gói kết quả cuối từ Kaggle Notebook Output về máy.

## Chặng 1 — Dữ liệu

- [x] Dataset public đã được chuẩn bị thành công bằng `--source` trên Kaggle.
- [x] `dataset_version` bắt đầu bằng `d10-`.
- [x] Split có đúng 1422/276/311 ảnh.
- [x] Box từng lớp khớp bảng trong README.
- [x] `dataset_summary()` xác nhận không có session leakage.
- [x] Đã xem trực quan ít nhất 6 ảnh ground truth.
- [x] Kết quả dữ liệu đã được lưu trong lịch sử commit của repository.

## Chặng 2–3 — E0 và pipeline huấn luyện

- [x] `work/e0.jsonl` có đủ 311 `image_id` không trùng trong Kaggle runtime.
- [x] `report/evidence/e0.json` có `split: test`, `score_version: 1` trong Kaggle runtime.
- [x] Đã lưu 5 output thô E0 để phân tích lỗi trong output notebook.
- [x] YOLO → JSON sắp ổn định theo `(y1, x1)`.
- [x] Kiểm tra một ảnh cho đúng 220 visual token.
- [x] Decode `labels != -100` ra đúng JSON assistant và token kết thúc.
- [x] Collator cat `pixel_values`/`image_grid_thw`, không pad ảnh.
- [x] Kết quả E0/mask loss đã được lưu trong lịch sử commit của repository.

## Chặng 4 — E1, E2, E3

- [x] E1 khớp 252 LoRA modules và hoàn tất 300 optimizer steps.
- [x] E2 khớp 252 LoRA modules, projector mở float32.
- [x] E3 khớp 348 LoRA modules, projector mở float32.
- [x] Loss cả ba setup hữu hạn.
- [x] E2/E3 đã lưu và nạp lại `projector.pt`.
- [x] Mỗi `run.json` có tham số trainable, VRAM đỉnh và phút chạy thật.
- [x] `e1.json`, `e2.json`, `e3.json` không thiếu prediction.
- [x] E1 có `parse_valid_rate >= 0.95` (thực tế 0.994).
- [x] F1(E1) − F1(E0) ≥ 0.10 (thực tế 0.1909).
- [x] Evidence đã được tải từ Kaggle; checkpoint/trọng số không được đưa vào Git.
- [x] Kết quả E1–E3 đã được lưu trong commit hoàn thiện bài nộp.

## Chặng 5 — E4

- [x] Chỉ chọn một biến: `data_size`.
- [x] Hai nhánh E4 giữ nguyên mọi biến còn lại.
- [x] Có hai evidence E4 đặt tên đúng quy ước.
- [x] Câu 10 thảo luận độ chồng lấn CI95, không chỉ so F1 điểm.
- [x] Kết quả E4 đã được lưu trong commit hoàn thiện bài nộp.

## Báo cáo và nghiệm thu

- [x] `REPORT.md` có họ tên và `dataset_version` thật.
- [x] Bảng E0–E4 lấy từ `report/evidence/*.json`.
- [x] Đã trả lời đủ 10 câu bằng số liệu của chính lần chạy này.
- [x] Không còn chuỗi `<điền>` trong `REPORT.md`.
- [x] Có 6 ảnh so sánh GT/prediction trong `report/figures/` và được dẫn trong báo cáo.
- [x] `git rev-list --count HEAD` trả về ít nhất 5.
- [x] Git không track `data/`, `work/`, `.env`, `*.pt`, `*.bin`, `*.safetensors`, `*.ckpt`.
- [x] Các test chức năng đạt; test template yêu cầu còn `<điền>` được loại sau khi báo cáo hoàn tất.
- [x] `python scripts/check_submission.py` báo `0 FAIL`.
- [x] Đã push lên GitHub; repository được xác nhận ở chế độ Public.

## Bonus E5 (không bắt buộc)

- [x] Reward đạt 5/5 ca kiểm thử độc lập trên CPU.
- [ ] GRPO chạy 60 bước từ model SFT tốt nhất.
- [ ] Có `report/evidence/e5.json` và phân tích reward hacking/F1/CI95.
