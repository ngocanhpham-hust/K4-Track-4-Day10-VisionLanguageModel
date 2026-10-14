# Checklist tiến độ Lab Day 10

Đánh dấu `[x]` ngay sau khi có bằng chứng thật. Không đánh dấu các mục GPU chỉ
dựa trên code đã viết.

## Repository và môi trường

- [x] `origin` trỏ tới fork GitHub cá nhân.
- [ ] Repository GitHub đang ở chế độ Public.
- [x] Không ghi Kaggle token trong source/notebook.
- [x] Bộ test công cụ đề bài đạt `68 passed, 1 skipped` trước khi chạy dữ liệu.
- [x] Kaggle Settings đã bật GPU và Internet.
- [x] `nvidia-smi -L` nhận T4 và working directory nằm dưới `/kaggle/working`.
- [ ] Đã Save Version để giữ checkpoint/evidence trong Notebook Output.

## Chặng 1 — Dữ liệu

- [x] Dataset public đã được chuẩn bị thành công bằng `--source` trên Kaggle.
- [x] `dataset_version` bắt đầu bằng `d10-`.
- [x] Split có đúng 1422/276/311 ảnh.
- [x] Box từng lớp khớp bảng trong README.
- [x] `dataset_summary()` xác nhận không có session leakage.
- [x] Đã xem trực quan ít nhất 6 ảnh ground truth.
- [ ] Đã tạo commit mốc dữ liệu.

## Chặng 2–3 — E0 và pipeline huấn luyện

- [x] `work/e0.jsonl` có đủ 311 `image_id` không trùng trong Kaggle runtime.
- [x] `report/evidence/e0.json` có `split: test`, `score_version: 1` trong Kaggle runtime.
- [x] Đã lưu 5 output thô E0 để phân tích lỗi trong output notebook.
- [x] YOLO → JSON sắp ổn định theo `(y1, x1)`.
- [x] Kiểm tra một ảnh cho đúng 220 visual token.
- [x] Decode `labels != -100` ra đúng JSON assistant và token kết thúc.
- [x] Collator cat `pixel_values`/`image_grid_thw`, không pad ảnh.
- [ ] Đã tạo commit mốc E0/mask loss.

## Chặng 4 — E1, E2, E3

- [x] E1 khớp 252 LoRA modules và hoàn tất 300 optimizer steps.
- [ ] E2 khớp 252 LoRA modules, projector mở float32.
- [ ] E3 khớp 348 LoRA modules, projector mở float32.
- [ ] Loss cả ba setup hữu hạn.
- [ ] E2/E3 đã lưu và nạp lại `projector.pt`.
- [ ] Mỗi `run.json` có tham số trainable, VRAM đỉnh và phút chạy thật.
- [ ] `e1.json`, `e2.json`, `e3.json` không thiếu prediction.
- [x] E1 có `parse_valid_rate >= 0.95` (thực tế 0.994).
- [x] F1(E1) − F1(E0) ≥ 0.10 (thực tế 0.1909).
- [ ] Đã sao lưu artifact vào Kaggle Notebook Output, không đưa vào Git.
- [ ] Đã tạo commit mốc E1–E3.

## Chặng 5 — E4

- [x] Chỉ chọn một biến: `data_size`.
- [ ] Hai nhánh E4 giữ nguyên mọi biến còn lại.
- [ ] Có hai evidence E4 đặt tên đúng quy ước.
- [ ] Câu 10 thảo luận độ chồng lấn CI95, không chỉ so F1 điểm.
- [ ] Đã tạo commit mốc E4.

## Báo cáo và nghiệm thu

- [x] `REPORT.md` có họ tên và `dataset_version` thật.
- [ ] Bảng E0–E4 lấy từ `report/evidence/*.json`.
- [ ] Đã trả lời đủ 10 câu bằng số liệu của chính lần chạy này.
- [x] Không còn chuỗi `<điền>` trong `REPORT.md`.
- [ ] Có ít nhất 6 ảnh GT/prediction trong `report/figures/` và được dẫn trong báo cáo (hiện đã khôi phục 6 ảnh GT; cell cuối sẽ tạo ảnh so sánh).
- [ ] `git rev-list --count HEAD` trả về ít nhất 5.
- [ ] Git không track `data/`, `work/`, `.env`, `*.pt`, `*.bin`, `*.safetensors`, `*.ckpt`.
- [ ] `python -m pytest -q` đạt.
- [ ] `python scripts/check_submission.py` báo `0 FAIL`.
- [ ] Đã push lên GitHub và kiểm tra URL bằng cửa sổ ẩn danh.

## Bonus E5 (không bắt buộc)

- [x] Reward đạt 5/5 ca kiểm thử độc lập trên CPU.
- [ ] GRPO chạy 60 bước từ model SFT tốt nhất.
- [ ] Có `report/evidence/e5.json` và phân tích reward hacking/F1/CI95.
