# Checklist tiến độ Lab Day 10

Đánh dấu `[x]` ngay sau khi có bằng chứng thật. Không đánh dấu các mục GPU chỉ
dựa trên code đã viết.

## Repository và môi trường

- [x] `origin` trỏ tới fork GitHub cá nhân.
- [ ] Repository GitHub đang ở chế độ Public.
- [x] Không ghi Kaggle token trong source/notebook.
- [x] Bộ test công cụ đề bài đạt `68 passed, 1 skipped` trước khi chạy dữ liệu.
- [ ] Kaggle Settings đã bật GPU và Internet.
- [ ] `nvidia-smi -L` nhận P100/T4 và working directory nằm dưới `/kaggle/working`.
- [ ] Đã Save Version để giữ checkpoint/evidence trong Notebook Output.

## Chặng 1 — Dữ liệu

- [ ] `python scripts/prepare_data.py --kaggle` chạy thành công.
- [ ] `dataset_version` bắt đầu bằng `d10-`.
- [ ] Split có đúng 1422/276/311 ảnh.
- [ ] Box từng lớp khớp bảng trong README.
- [ ] `dataset_summary()` xác nhận không có session leakage.
- [ ] Đã xem trực quan ít nhất 6 ảnh ground truth.
- [ ] Đã tạo commit mốc dữ liệu.

## Chặng 2–3 — E0 và pipeline huấn luyện

- [ ] `work/e0.jsonl` có đủ 311 `image_id` không trùng.
- [ ] `report/evidence/e0.json` có `split: test`, `score_version: 1`.
- [ ] Đã lưu 5 output thô E0 để phân tích lỗi.
- [ ] YOLO → JSON sắp ổn định theo `(y1, x1)`.
- [ ] Kiểm tra một ảnh cho đúng 220 visual token.
- [ ] Decode `labels != -100` ra đúng JSON assistant và token kết thúc.
- [ ] Collator cat `pixel_values`/`image_grid_thw`, không pad ảnh.
- [ ] Đã tạo commit mốc E0/mask loss.

## Chặng 4 — E1, E2, E3

- [ ] E1 khớp 252 LoRA modules và hoàn tất 300 optimizer steps.
- [ ] E2 khớp 252 LoRA modules, projector mở float32.
- [ ] E3 khớp 348 LoRA modules, projector mở float32.
- [ ] Loss cả ba setup hữu hạn.
- [ ] E2/E3 đã lưu và nạp lại `projector.pt`.
- [ ] Mỗi `run.json` có tham số trainable, VRAM đỉnh và phút chạy thật.
- [ ] `e1.json`, `e2.json`, `e3.json` không thiếu prediction.
- [ ] E1 có `parse_valid_rate >= 0.95`.
- [ ] F1(E1) − F1(E0) ≥ 0.10.
- [ ] Đã sao lưu artifact lên Drive, không đưa vào Git.
- [ ] Đã tạo commit mốc E1–E3.

## Chặng 5 — E4

- [ ] Chỉ chọn một biến: `rank`, `data_size` hoặc `coord_scale`.
- [ ] Hai nhánh E4 giữ nguyên mọi biến còn lại.
- [ ] Có hai evidence E4 đặt tên đúng quy ước.
- [ ] Câu 10 thảo luận độ chồng lấn CI95, không chỉ so F1 điểm.
- [ ] Đã tạo commit mốc E4.

## Báo cáo và nghiệm thu

- [ ] `REPORT.md` có họ tên và `dataset_version` thật.
- [ ] Bảng E0–E4 lấy từ `report/evidence/*.json`.
- [ ] Đã trả lời đủ 10 câu bằng số liệu của chính lần chạy này.
- [ ] Không còn chuỗi `<điền>` trong `REPORT.md`.
- [ ] Có ít nhất 6 ảnh GT/prediction trong `report/figures/` và được dẫn trong báo cáo.
- [ ] `git rev-list --count HEAD` trả về ít nhất 5.
- [ ] Git không track `data/`, `work/`, `.env`, `*.pt`, `*.bin`, `*.safetensors`, `*.ckpt`.
- [ ] `python -m pytest -q` đạt.
- [ ] `python scripts/check_submission.py` báo `0 FAIL`.
- [ ] Đã push lên GitHub và kiểm tra URL bằng cửa sổ ẩn danh.

## Bonus E5 (không bắt buộc)

- [x] Reward đạt 5/5 ca kiểm thử độc lập trên CPU.
- [ ] GRPO chạy 60 bước từ model SFT tốt nhất.
- [ ] Có `report/evidence/e5.json` và phân tích reward hacking/F1/CI95.
