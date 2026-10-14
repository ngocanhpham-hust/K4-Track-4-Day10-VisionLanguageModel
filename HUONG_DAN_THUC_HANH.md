# Hướng dẫn thực hành Lab Day 10

`lab_day10.ipynb` là notebook Kaggle tự chứa: cell đầu tự clone repository và
toàn bộ hàm lab đã được nhúng trực tiếp. `lab_day10_solution.py` là bản module
tương đương để kiểm thử trong repository. Không tự điền số mẫu vào `REPORT.md`:
mọi F1, CI95, VRAM và thời gian phải lấy từ lần chạy thật.

## 1. Fork, clone và kiểm tra remote

1. Fork repository gốc sang tài khoản GitHub cá nhân, để chế độ **Public**.
2. Clone fork cá nhân, không clone repository của tổ chức để làm bài.
3. Trong thư mục dự án, kiểm tra:

```bash
git remote -v
git status
python -m pytest -q
```

Kỳ vọng ban đầu: `68 passed, 1 skipped`. URL `origin` phải chứa username GitHub
của bạn. Tuyệt đối không commit Kaggle token, `data/`, `work/` hoặc trọng số.

## 2. Mở notebook trên Kaggle

Import `lab_day10.ipynb` vào Kaggle, mở **Notebook Settings**, chọn
**Accelerator → GPU** và bật **Internet → On**, rồi chạy từ trên xuống. Cell đầu
tự clone repository vào `/kaggle/working`; không cần upload thêm file.

Notebook đã có sẵn code để:

- cài đúng phiên bản trong `requirements-colab.txt` và chạy test;
- tự lấy public dataset bằng `kagglehub`, không cần API secret;
- giữ checkpoint/evidence trong `/kaggle/working` để xuất cùng Notebook Output;
- tải/chia dữ liệu, kiểm tra session leakage và vẽ 6 ảnh GT;
- chạy E0;
- kiểm tra mask loss, chuỗi target và 220 visual token;
- train, lưu và chấm E1–E3;
- chạy E4 `data_size` và kiểm tra reward E5.

Sau khi chuẩn bị dữ liệu, kết quả sau phải khớp:

```python
from lab_day10_solution import dataset_summary
dataset_summary()
```

| Split | Ảnh | pothole | crack | manhole |
|---|---:|---:|---:|---:|
| train | 1422 | 1074 | 1750 | 691 |
| val | 276 | 66 | 470 | 103 |
| test | 311 | 121 | 299 | 163 |

Commit gợi ý sau chặng dữ liệu:

```bash
git add lab_day10.ipynb
git commit -m "Verify deterministic session splits"
```

## 3. E0 zero-shot

Notebook gọi tương đương:

```python
from lab_day10_solution import load_model_and_processor, run_inference

model, processor = load_model_and_processor()
run_inference(
    model,
    processor,
    split="test",
    output_path="work/e0.jsonl",
    batch_size=4,
    resume=True,
)
```

`resume=True` cho phép tiếp tục nếu Kaggle session bị ngắt. Nếu muốn chạy lại bằng model
khác, xóa đúng file `work/e0.jsonl` hoặc đặt `resume=False`, nếu không các ảnh đã
có sẽ bị bỏ qua.

Chấm E0:

```bash
python scripts/score.py \
  --pred work/e0.jsonl \
  --name E0 \
  --out report/evidence/e0.json
```

Đọc `parse_valid_rate`, F1 và ít nhất 5 output thô để trả lời Câu 2. Commit
`lab_day10.ipynb` và `report/evidence/e0.json`, không commit `work/e0.jsonl`.

## 4. Kiểm tra chat template và loss mask

Ô mục 4 của notebook phải in:

- `vision tokens: 220`;
- phần decode của `labels != -100` là JSON đích và token kết thúc assistant;
- `labels` ở phần ảnh/prompt và vùng padding đều bằng `-100`.

Nếu assertion tiền tố chat template lỗi, dừng lại; không train với nhãn bị lệch.

## 5. Train E1, E2 và E3

Chạy **mỗi setup trong một runtime sạch**. Đổi biến sau ở mục 5 notebook:

```python
SETUP = "E1"  # sau đó E2, rồi E3
```

Các tín hiệu đúng:

- E1/E2 khớp 252 module LoRA;
- E3 khớp 348 module LoRA;
- E2/E3 có projector được mở ở float32;
- loss luôn hữu hạn và nhìn chung giảm;
- `work/eN/adapter/` tồn tại;
- E2/E3 có thêm `work/eN/projector.pt`;
- `work/eN/run.json` có `trainable_params`, `peak_vram_gb`, `train_minutes`.

Không dùng `model.save_pretrained()` như cách duy nhất cho E2/E3. Hàm
`train_sft()` đã lưu projector riêng; hàm `load_trained_model()` sẽ từ chối suy
luận E2/E3 nếu thiếu file đó.

Artifact được ghi vào
`/kaggle/working/K4-Track-4-Day10-VisionLanguageModel/work/`. Chọn **Save
Version** để giữ chúng trong Notebook Output. Adapter/projector chỉ để trong
Output hoặc `work/`, không đưa vào Git.

Sau train, ô mục 7 sinh `work/eN.jsonl` và tự truyền metadata thật từ
`work/eN/run.json` cho scorer. Kiểm tra `missing_predictions == 0` và
`parse_valid_rate >= 0.95` cho E1.

Commit gợi ý sau khi đã có đủ E1–E3:

```bash
git add lab_day10.ipynb report/evidence/e1.json report/evidence/e2.json report/evidence/e3.json
git commit -m "Complete controlled E1 E2 E3 experiments"
```

## 6. E4 đơn biến

Khuyến nghị chọn `data_size`, vì không phải đổi prompt/scorer. Xác định setup
tốt nhất trong E1–E3, rồi giữ nguyên setup, rank, LR, batch, số bước và seed.

Ví dụ nếu E1 tốt nhất, train nhánh 10%:

```python
from lab_day10_solution import train_experiment, run_inference, score_experiment

model_e4, processor_e4, _ = train_experiment(
    "E1", train_fraction=0.1, output_dir="work/e4_data_0.1"
)
run_inference(
    model_e4,
    processor_e4,
    output_path="work/e4_data_0.1.jsonl",
    resume=True,
)
score_experiment(
    "work/e4_data_0.1.jsonl",
    "E4_data_0.1",
    "report/evidence/e4_data_0.1.json",
    run_meta_path="work/e4_data_0.1/run.json",
)
```

Nhánh 100% chính là prediction của setup tốt nhất nếu tất cả cấu hình còn lại
giống hệt; chấm lại file đó với tên E4:

```python
score_experiment(
    "work/e1.jsonl",  # đổi theo setup tốt nhất
    "E4_data_1.0",
    "report/evidence/e4_data_1.0.json",
    run_meta_path="work/e1/run.json",
)
```

So sánh cả F1 và hai khoảng CI95. Nếu CI chồng lấn, không tuyên bố chắc chắn một
cấu hình tốt hơn; chỉ nói dữ liệu hiện có chưa đủ bằng chứng phân biệt.

## 7. Xuất ít nhất 6 ảnh minh họa

Sau khi chọn model tốt nhất:

```python
from lab_day10_solution import export_prediction_visualizations

paths = export_prediction_visualizations(
    "work/e1.jsonl",  # đổi theo model tốt nhất
    "report/figures",
    n=6,
)
paths
```

Ảnh dùng box xanh cho GT và box đỏ cho prediction. Chèn chúng vào Câu 9 của
`REPORT.md`, ví dụ `![Ví dụ lỗi](report/figures/01_....jpg)`.

## 8. E5 tùy chọn

Luôn chạy kiểm thử reward trước:

```python
from lab_day10_solution import check_reward_cases
check_reward_cases()
```

Kết quả phải là 1.0, 0.0, 0.2, 0.6, 1.0 cho năm ca đề bài. Sau đó nạp model
tốt nhất, tạo dataset và trainer:

```python
from lab_day10_solution import (
    build_grpo_dataset, create_grpo_trainer, load_trained_model,
)

# Đổi E1/work/e1 theo setup tốt nhất. is_trainable=True để tiếp tục cập nhật LoRA.
model, processor = load_trained_model(
    "work/e1", setup="E1", is_trainable=True
)
grpo_data = build_grpo_dataset(processor)
trainer = create_grpo_trainer(model, processor, grpo_data, output_dir="work/e5")
trainer.train()
trainer.save_model("work/e5/adapter")
```

GRPO tạo bốn completion mỗi ảnh nên nhạy với VRAM hơn SFT. Nếu GPU hết bộ nhớ,
E5 là phần thưởng và có thể bỏ; không thay đổi E1–E4 chỉ để ép E5 chạy.

## 9. Điền báo cáo từ evidence thật

Lấy nhanh các cột cần thiết:

```python
from lab_day10_solution import evidence_summary
evidence_summary()
```

Điền họ tên, `dataset_version`, bảng số liệu và 10 câu trả lời. Xóa toàn bộ chuỗi
`<điền>`. Câu trả lời so sánh phải dùng F1/CI95/VRAM/phút của chính bạn.

## 10. Nghiệm thu và nộp

```bash
python -m pytest -q
python scripts/check_submission.py
git status --short
git ls-files | grep -E '^(data/|work/)|(^|/)\.env$|\.(pt|bin|safetensors|ckpt)$' || true
git rev-list --count HEAD
```

Chỉ nộp khi checker báo `0 FAIL`. Kiểm tra `git diff` trước khi commit cuối:

```bash
git add lab_day10.ipynb lab_day10_solution.py HUONG_DAN_THUC_HANH.md REPORT.md report/
git commit -m "Complete Lab Day 10 report and evidence"
git push origin main
```

Cuối cùng mở URL GitHub khi chưa đăng nhập/ở cửa sổ ẩn danh để xác nhận repository
thật sự Public và các evidence/ảnh/report đều truy cập được.
