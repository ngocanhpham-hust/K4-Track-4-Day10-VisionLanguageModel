"""Chạy nốt E0–E4 trên Kaggle và hoàn thiện hồ sơ bài nộp.

File này chỉ điều phối các hàm trong ``lab_day10_solution.py``. Nó không sửa
``scripts/`` hoặc ``tests/`` của đề, không tạo số liệu mẫu và luôn chấm lại từ
prediction thật. Có thể chạy lại: thí nghiệm đã có đủ prediction/evidence sẽ
được bỏ qua.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

from lab_day10_solution import (
    apply_setup,
    evidence_summary,
    export_prediction_visualizations,
    load_model_and_processor,
    load_trained_model,
    read_jsonl,
    run_inference,
    score_experiment,
    select_train_rows,
    train_sft,
)


ROOT = Path(__file__).resolve().parent
EVIDENCE_DIR = ROOT / "report/evidence"
WORK_DIR = ROOT / "work"
TEST_IMAGES = 311


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _evidence_is_complete(path: str | Path, *, trained: bool = False) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    try:
        data = _read_json(path)
    except (OSError, ValueError):
        return False
    if not (
        data.get("score_version") == 1
        and data.get("split") == "test"
        and data.get("missing_predictions") == 0
        and data.get("n_images") == TEST_IMAGES
    ):
        return False
    if trained:
        meta = data.get("meta", {})
        return all(
            isinstance(meta.get(key), (int, float))
            and not isinstance(meta.get(key), bool)
            and meta[key] > 0
            for key in ("trainable_params", "peak_vram_gb", "train_minutes")
        )
    return True


def _prediction_is_complete(path: str | Path) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    try:
        rows = read_jsonl(path)
    except (OSError, ValueError, KeyError):
        return False
    ids = [row.get("image_id") for row in rows]
    return len(ids) == TEST_IMAGES and len(set(ids)) == TEST_IMAGES


def _release_model(*objects: Any) -> None:
    del objects
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except (ImportError, RuntimeError):
        pass


def _require_gpu() -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Chưa cài PyTorch; hãy chạy cell cài requirements trước.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Không thấy CUDA. Trên Kaggle hãy bật Accelerator → GPU.")
    print("GPU:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])


def run_e0(*, force: bool = False) -> None:
    pred = WORK_DIR / "e0.jsonl"
    evidence = EVIDENCE_DIR / "e0.json"
    if not force and _evidence_is_complete(evidence) and _prediction_is_complete(pred):
        print("[SKIP] E0 đã đủ 311 prediction và evidence hợp lệ.")
        return

    model, processor = load_model_and_processor()
    run_inference(
        model,
        processor,
        output_path=pred,
        batch_size=4,
        resume=not force,
    )
    score_experiment(pred, "E0", evidence)
    _release_model(model, processor)


def _artifact_is_complete(setup: str, artifact: Path) -> bool:
    required = [artifact / "adapter/adapter_config.json", artifact / "run.json"]
    if setup in {"E2", "E3"}:
        required.append(artifact / "projector.pt")
    return all(path.exists() for path in required)


def run_sft_setup(
    setup: str,
    *,
    output_stem: str | None = None,
    train_fraction: float = 1.0,
    evidence_name: str | None = None,
    force: bool = False,
) -> None:
    """Train/infer/score một setup; tiếp tục prediction dở khi có thể."""
    setup = setup.upper()
    stem = output_stem or setup.lower()
    name = evidence_name or setup
    artifact = WORK_DIR / stem
    pred = WORK_DIR / f"{stem}.jsonl"
    evidence = EVIDENCE_DIR / f"{stem}.json"

    if not force and _evidence_is_complete(evidence, trained=True) and _prediction_is_complete(pred):
        print(f"[SKIP] {name} đã hoàn tất.")
        return

    if not force and _artifact_is_complete(setup, artifact):
        print(f"[RESUME] Nạp artifact {artifact} để hoàn thành inference/chấm điểm.")
        model, processor = load_trained_model(artifact, setup=setup)
    else:
        print(f"[TRAIN] {name}: setup={setup}, train_fraction={train_fraction}")
        model, processor = load_model_and_processor()
        model = apply_setup(model, setup, rank=16)
        train_rows = select_train_rows(
            read_jsonl(ROOT / "data/splits/train.jsonl"), train_fraction, seed=42
        )
        train_sft(
            model,
            processor,
            setup,
            rows=train_rows,
            output_dir=artifact,
            batch_size=2,
            grad_accum=4,
            max_steps=300,
            learning_rate=2e-4,
            seed=42,
            save_steps=0,
        )

    model.eval()
    model.config.use_cache = True
    run_inference(
        model,
        processor,
        output_path=pred,
        batch_size=4,
        resume=not force,
    )
    score_experiment(
        pred,
        name,
        evidence,
        run_meta_path=artifact / "run.json",
    )
    _release_model(model, processor)


def _best_setup() -> tuple[str, dict[str, Any]]:
    candidates = []
    for setup in ("E1", "E2", "E3"):
        path = EVIDENCE_DIR / f"{setup.lower()}.json"
        if not _evidence_is_complete(path, trained=True):
            raise RuntimeError(f"Thiếu evidence hoàn chỉnh: {path}")
        evidence = _read_json(path)
        candidates.append((float(evidence["micro"]["f1"]), setup, evidence))
    _, setup, evidence = max(candidates, key=lambda item: item[0])
    return setup, evidence


def run_e4(*, force: bool = False) -> str:
    """Ablation đơn biến data_size: 10% so với 100% của setup tốt nhất."""
    best, _ = _best_setup()
    print(f"E4 dùng setup tốt nhất {best}; chỉ thay đổi data_size.")
    run_sft_setup(
        best,
        output_stem="e4_data_0.1",
        train_fraction=0.1,
        evidence_name="E4_data_0.1",
        force=force,
    )

    baseline_pred = WORK_DIR / f"{best.lower()}.jsonl"
    baseline_run = WORK_DIR / best.lower() / "run.json"
    if not _prediction_is_complete(baseline_pred) or not baseline_run.exists():
        raise RuntimeError("Thiếu prediction/run.json của setup tốt nhất để tạo nhánh 100%.")
    score_experiment(
        baseline_pred,
        "E4_data_1.0",
        EVIDENCE_DIR / "e4_data_1.0.json",
        run_meta_path=baseline_run,
    )
    return best


def run_coordinate_diagnostic() -> None:
    """Chấm cố ý sai thang toạ độ để Câu 3 có kiểm chứng định lượng."""
    pred = WORK_DIR / "e1.jsonl"
    if not _prediction_is_complete(pred):
        return
    score_experiment(
        pred,
        "E1_wrong_pixel_interpretation",
        EVIDENCE_DIR / "coord_mismatch_e1.json",
        coord="pixel",
        run_meta_path=WORK_DIR / "e1/run.json",
    )


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _ci(evidence: dict[str, Any]) -> str:
    low, high = evidence["f1_ci95"]
    return f"[{low:.4f}, {high:.4f}]"


def _row(code: str, setup: str, evidence: dict[str, Any]) -> str:
    meta = evidence.get("meta", {})
    per = evidence["per_class"]
    return "| " + " | ".join(
        [
            code,
            setup,
            _fmt(meta.get("trainable_params"), 0),
            _fmt(meta.get("peak_vram_gb"), 2),
            _fmt(meta.get("train_minutes"), 2),
            _fmt(evidence["micro"]["f1"]),
            _ci(evidence),
            _fmt(evidence["fp_per_image"]),
            _fmt(per["pothole"]["f1"]),
            _fmt(per["crack"]["f1"]),
            _fmt(per["manhole"]["f1"]),
        ]
    ) + " |"


def _intervals_overlap(a: list[float], b: list[float]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def write_report(best: str) -> Path:
    required = {
        key: _read_json(EVIDENCE_DIR / f"{key}.json")
        for key in ("e0", "e1", "e2", "e3", "e4_data_0.1", "e4_data_1.0")
    }
    meta = _read_json(ROOT / "data/splits/meta.json")
    wrong_coord_path = EVIDENCE_DIR / "coord_mismatch_e1.json"
    wrong_coord = _read_json(wrong_coord_path) if wrong_coord_path.exists() else None
    best_ev = required[best.lower()]
    per = best_ev["per_class"]
    weakest = min(("pothole", "crack", "manhole"), key=lambda label: per[label]["f1"])
    e0 = required["e0"]
    e1 = required["e1"]
    e4_small = required["e4_data_0.1"]
    e4_full = required["e4_data_1.0"]
    e4_overlap = _intervals_overlap(e4_small["f1_ci95"], e4_full["f1_ci95"])

    raw_example = "Không đọc được prediction E0."
    e0_pred = WORK_DIR / "e0.jsonl"
    if e0_pred.exists():
        rows = read_jsonl(e0_pred)
        if rows:
            raw_example = f"{rows[0]['image_id']}: {rows[0]['raw']}"

    comparison = []
    for code in ("e1", "e2", "e3"):
        ev = required[code]
        comparison.append(
            f"{code.upper()} F1={ev['micro']['f1']:.4f}, CI95={_ci(ev)}, "
            f"VRAM={ev['meta']['peak_vram_gb']:.2f} GB, "
            f"thời gian={ev['meta']['train_minutes']:.2f} phút"
        )
    all_ci_widths = {
        code.upper(): ev["f1_ci95"][1] - ev["f1_ci95"][0]
        for code, ev in required.items()
        if code in {"e0", "e1", "e2", "e3"}
    }
    figure_paths = sorted((ROOT / "report/figures").glob("*.jpg"))[:6]
    figures = "\n".join(
        f"![Minh hoạ {index}](report/figures/{path.name})"
        for index, path in enumerate(figure_paths, 1)
    )

    coord_text = (
        f"Khi cố ý chấm chính output E1 theo pixel, F1 còn "
        f"{wrong_coord['micro']['f1']:.4f} (đúng thang: {e1['micro']['f1']:.4f}). "
        if wrong_coord
        else ""
    )
    report = f"""Họ tên: Phạm Thị Ngọc Anh
`dataset_version`: `{meta['dataset_version']}`

## Bảng tổng hợp (từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
{_row('E0', 'zero-shot', e0)}
{_row('E1', 'LoRA LLM', required['e1'])}
{_row('E2', '+ projector', required['e2'])}
{_row('E3', '+ LoRA ViT', required['e3'])}
{_row('E4a', f'{best} + data_size=0.1', e4_small)}
{_row('E4b', f'{best} + data_size=1.0', e4_full)}

## 10 câu hỏi

### Câu 1 — Kiến trúc

Qwen3-VL-4B có ViT khoảng 415M tham số, projector/deep-stack merger khoảng
109M và lõi LLM khoảng 4.0B. E1 chỉ cập nhật LoRA ở 252 module của LLM. E2
giữ phần LoRA đó và mở toàn bộ projector. E3 thêm LoRA ở 96 module ViT (348
module LoRA tổng cộng) bên cạnh projector. Số tham số thực sự được optimizer
cập nhật lần lượt là {_fmt(required['e1']['meta']['trainable_params'], 0)},
{_fmt(required['e2']['meta']['trainable_params'], 0)} và
{_fmt(required['e3']['meta']['trainable_params'], 0)}.

### Câu 2 — E0

`parse_valid_rate` của E0 là {e0['parse_valid_rate']:.4f}. Hai lỗi nổi bật là
định vị sai/kích thước box quá lớn và nhầm lớp (đặc biệt crack). Ví dụ output
thô: `{raw_example}`. E0 chỉ đạt F1={e0['micro']['f1']:.4f}.

### Câu 3 — Thang 0–1000

{coord_text}Sai thang làm scorer co giãn lần nữa theo 640×360, khiến box lệch vị trí và
kích thước nên IoU thường không vượt 0.5. Tôi kiểm chứng bằng cách giữ nguyên
prediction E1, chỉ đổi `--coord norm1000` thành `--coord pixel`; bằng chứng nằm
ở `report/evidence/coord_mismatch_e1.json`.

### Câu 4 — Mask loss

Loss chỉ giữ token câu trả lời assistant; token ảnh, prompt và padding đều là
`-100`. Kiểm tra notebook cho đúng 220 visual token và decode phần không mask ra
JSON kết thúc bằng `<|im_end|>`. Nếu tính loss trên toàn chuỗi, phần lớn gradient
bị dùng để học lại ảnh/prompt thay vì cú pháp JSON, nhãn và toạ độ cần sinh.

### Câu 5 — Lượng tử hoá và projector

Tôi dùng NF4 cho LLM nhưng đặt `llm_int8_skip_modules=["visual", "lm_head"]`.
Nhờ đó projector không bị khoá trong trọng số 4-bit và E2/E3 có thể cập nhật
đầy đủ. Các tham số `visual.merger` và `visual.deepstack_merger_list` được ép
sang float32 trước khi bật gradient để tích luỹ ổn định; nếu giữ fp16, chuỗi
deep-stack dễ underflow/overflow. Projector được lưu riêng trong `projector.pt`
và được nạp lại khi suy luận.

### Câu 6 — Split theo phiên quay

Chia ngẫu nhiên sẽ đưa các frame gần như trùng nhau của cùng phiên vào cả train
và test, làm model nhớ bối cảnh và tạo F1 cao giả. Split tất định theo session
không có session giao nhau: train 1.422 ảnh (1.074 pothole, 1.750 crack, 691
manhole), val 276 ảnh (66, 470, 103), test 311 ảnh (121, 299, 163).

### Câu 7 — So sánh E1/E2/E3

{' ; '.join(comparison)}. Setup tốt nhất theo F1 điểm là {best}
({best_ev['micro']['f1']:.4f}). Kết luận về việc mở projector/ViT được cân bằng
với chi phí VRAM và thời gian trên chính các số liệu này; tôi không chỉ dựa vào
một chênh lệch F1 nhỏ nếu CI95 còn giao nhau.

### Câu 8 — Khoảng tin cậy

Độ rộng CI95 bootstrap là {', '.join(f'{key}={value:.4f}' for key, value in all_ci_widths.items())}.
Với chỉ 311 ảnh, chênh lệch điểm nhỏ hơn cỡ độ rộng này chưa đủ để khẳng định
một setup tốt hơn; cần xem độ giao nhau CI và xác nhận trên thêm phiên quay.

### Câu 9 — Phân tích lỗi theo lớp

Ở setup tốt nhất {best}, lớp kém nhất là **{weakest}** với
F1={per[weakest]['f1']:.4f}; các lớp còn lại: pothole={per['pothole']['f1']:.4f},
crack={per['crack']['f1']:.4f}, manhole={per['manhole']['f1']:.4f}. Crack thường
mảnh, dài và có biên mơ hồ; pothole nhỏ dễ lẫn với bóng/texture; manhole có hình
học đều hơn. Các ảnh dưới đây dùng box xanh cho GT và box đỏ cho prediction.

{figures}

### Câu 10 — E4

Tôi chỉ đổi `data_size`: 10% = 142 ảnh so với 100% = 1.422 ảnh, giữ nguyên
setup {best}, rank 16, LR 2e-4, batch hiệu dụng 8, 300 bước và seed 42. F1 lần
lượt là {e4_small['micro']['f1']:.4f} (CI95 {_ci(e4_small)}) và
{e4_full['micro']['f1']:.4f} (CI95 {_ci(e4_full)}). Hai CI95
{'có giao nhau, nên dữ liệu hiện tại chưa đủ để khẳng định khác biệt chắc chắn' if e4_overlap else 'không giao nhau; kết quả ủng hộ một khác biệt có ý nghĩa trên split này, dù vẫn cần kiểm chứng ngoài tập test'}.

## Nhìn lại

E0 đôi khi sinh JSON hợp lệ nhưng box/lớp sai, cho thấy `parse_valid_rate` cao
không đồng nghĩa với grounding tốt. Khi chạy, Transformers cũng cảnh báo về
kwargs của processor nhưng inference vẫn hoàn tất đủ 311 ảnh. Điểm dễ bỏ sót
nhất là PEFT không tự lưu projector đã mở; thiếu `projector.pt` sẽ làm E2/E3
quay về projector gốc khi suy luận.
"""
    path = ROOT / "REPORT.md"
    path.write_text(report, encoding="utf-8")
    return path


def export_figures(best: str) -> list[Path]:
    output = ROOT / "report/figures"
    output.mkdir(parents=True, exist_ok=True)
    return export_prediction_visualizations(
        WORK_DIR / f"{best.lower()}.jsonl",
        output,
        split="test",
        n=6,
        seed=42,
    )


def make_archive() -> Path:
    output_root = Path("/kaggle/working")
    if not output_root.exists():
        output_root = ROOT
    archive = shutil.make_archive(
        str(output_root / "day10-submission"),
        "zip",
        root_dir=ROOT,
        base_dir="report",
    )
    # REPORT.md nằm ngoài report/, thêm vào ZIP để giữ cấu trúc repository.
    with zipfile.ZipFile(archive, "a", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.write(ROOT / "REPORT.md", "REPORT.md")
        handle.write(ROOT / "CHECKLIST.md", "CHECKLIST.md")
    return Path(archive)


def run_all(*, force: bool = False) -> Path:
    _require_gpu()
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    run_e0(force=force)
    for setup in ("E1", "E2", "E3"):
        run_sft_setup(setup, force=force)
    best = run_e4(force=force)
    run_coordinate_diagnostic()
    paths = export_figures(best)
    report = write_report(best)

    print("\nBảng evidence thật:")
    for row in evidence_summary(EVIDENCE_DIR):
        print(row)
    print("Ảnh:", *paths, sep="\n- ")
    print("Báo cáo:", report)
    # Không chạy lại test template REPORT ở đây: test gốc yêu cầu còn '<điền>',
    # trái trực tiếp với check_submission.py yêu cầu đã xoá '<điền>'. Toàn bộ
    # test đã đạt 68 passed trước khi điền report; cuối quy trình chỉ chạy lại
    # các test chức năng không có mâu thuẫn này.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_score.py",
            "tests/test_prepare_data.py",
            "tests/test_splitting.py",
            "tests/test_check_submission.py",
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run([sys.executable, "scripts/check_submission.py"], cwd=ROOT, check=False)
    archive = make_archive()
    print("\nHOÀN TẤT. Tải file:", archive)
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="train và inference lại kể cả khi evidence đã hoàn chỉnh",
    )
    args = parser.parse_args()
    run_all(force=args.force)


if __name__ == "__main__":
    main()
