"""Code thực hành cho Lab Day 10 (chạy các phần GPU trên Kaggle hoặc Colab).

File này không thay đổi công cụ chấm trong ``scripts/``. Các import nặng được đặt
trong hàm để vẫn có thể kiểm tra phần đổi nhãn và reward trên máy không có CUDA.
"""

from __future__ import annotations

import json
import math
import random
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, ImageDraw

from scripts.labcommon import CLASSES, read_jsonl, write_jsonl
from scripts.score import iou_xyxy, parse_output


ROOT = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

PROMPT_NORM1000 = (
    "Locate road damage in this image. "
    "Detect only objects of classes: pothole, crack, manhole. "
    "Output a JSON list of objects with label and bbox_2d coordinates "
    "normalized on a 0-1000 scale: "
    '[{"label": "...", "bbox_2d": [x1, y1, x2, y2]}]. '
    "If no damage is found, output []. Return only the JSON list."
)

PROMPT_PIXEL = (
    "Locate road damage in this image. "
    "Detect only objects of classes: pothole, crack, manhole. "
    "Output a JSON list of objects with label and bbox_2d coordinates "
    "in absolute image pixels: "
    '[{"label": "...", "bbox_2d": [x1, y1, x2, y2]}]. '
    "If no damage is found, output []. Return only the JSON list."
)

_LLM_TARGET = (
    r".*language_model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)
_VIT_TARGET = (
    r".*visual\.blocks\.\d+\."
    r"(?:attn\.(?:qkv|proj)|mlp\.(?:linear_fc1|linear_fc2))$"
)


def prompt_for_coord(coord: str) -> str:
    """Trả prompt ứng với quy ước toạ độ của thí nghiệm."""
    if coord == "norm1000":
        return PROMPT_NORM1000
    if coord == "pixel":
        return PROMPT_PIXEL
    raise ValueError("coord phải là 'norm1000' hoặc 'pixel'")


def _positive_box(x1: int, y1: int, x2: int, y2: int, maximum: int) -> list[int]:
    """Kẹp box vào miền hợp lệ và bảo đảm chiều rộng/cao ít nhất một đơn vị."""
    x1, y1 = max(0, min(maximum, x1)), max(0, min(maximum, y1))
    x2, y2 = max(0, min(maximum, x2)), max(0, min(maximum, y2))
    if x2 <= x1:
        if x1 == maximum:
            x1 = maximum - 1
        x2 = x1 + 1
    if y2 <= y1:
        if y1 == maximum:
            y1 = maximum - 1
        y2 = y1 + 1
    return [x1, y1, x2, y2]


def yolo_to_box(
    label: str,
    yolo: Sequence[float],
    *,
    coord: str = "norm1000",
    width: int | None = None,
    height: int | None = None,
) -> dict[str, Any]:
    """Đổi ``[xc, yc, w, h]`` chuẩn hoá sang JSON grounding.

    ``coord='norm1000'`` tạo số nguyên trong [0, 1000]. ``coord='pixel'``
    dùng kích thước ảnh thật và phục vụ ablation E4.
    """
    if label not in CLASSES:
        raise ValueError(f"nhãn không hợp lệ: {label!r}")
    if len(yolo) != 4 or not all(math.isfinite(float(v)) for v in yolo):
        raise ValueError(f"YOLO box phải gồm bốn số hữu hạn: {yolo!r}")
    xc, yc, bw, bh = (float(v) for v in yolo)
    if not all(0.0 <= v <= 1.0 for v in (xc, yc, bw, bh)):
        raise ValueError(f"YOLO box nằm ngoài [0, 1]: {yolo!r}")

    if coord == "norm1000":
        sx = sy = maximum = 1000
    elif coord == "pixel":
        if not width or not height:
            raise ValueError("coord='pixel' cần width và height dương")
        sx, sy, maximum = int(width), int(height), None
    else:
        raise ValueError("coord phải là 'norm1000' hoặc 'pixel'")

    x1 = int(round((xc - bw / 2) * sx))
    y1 = int(round((yc - bh / 2) * sy))
    x2 = int(round((xc + bw / 2) * sx))
    y2 = int(round((yc + bh / 2) * sy))
    if maximum is not None:
        box = _positive_box(x1, y1, x2, y2, maximum)
    else:
        x1, x2 = max(0, min(sx, x1)), max(0, min(sx, x2))
        y1, y2 = max(0, min(sy, y1)), max(0, min(sy, y2))
        if x2 <= x1:
            x1, x2 = (sx - 1, sx) if x1 == sx else (x1, x1 + 1)
        if y2 <= y1:
            y1, y2 = (sy - 1, sy) if y1 == sy else (y1, y1 + 1)
        box = [x1, y1, x2, y2]
    return {"label": label, "bbox_2d": box}


def objects_to_answer(
    objects: Sequence[dict[str, Any]],
    *,
    coord: str = "norm1000",
    width: int | None = None,
    height: int | None = None,
) -> str:
    """Đổi mọi object và sắp xếp ổn định theo ``(y1, x1)``."""
    boxes = [
        yolo_to_box(
            obj["label"], obj["yolo"], coord=coord, width=width, height=height
        )
        for obj in objects
    ]
    boxes.sort(key=lambda item: (item["bbox_2d"][1], item["bbox_2d"][0]))
    return json.dumps(boxes, ensure_ascii=False, separators=(",", ":"))


def image_path(entry: dict[str, Any], data_dir: str | Path = "data") -> Path:
    """Giải đường dẫn ảnh từ một record split."""
    path = Path(entry["image_path"])
    return path if path.is_absolute() else Path(data_dir) / path


def dataset_summary(splits_dir: str | Path = "data/splits") -> dict[str, Any]:
    """Đếm ảnh/box và xác nhận không có session rò giữa các split."""
    splits_dir = Path(splits_dir)
    summary: dict[str, Any] = {}
    owner: dict[str, str] = {}
    leaks: list[tuple[str, str, str]] = []
    for split in ("train", "val", "test"):
        rows = read_jsonl(splits_dir / f"{split}.jsonl")
        boxes = Counter(obj["label"] for row in rows for obj in row["objects"])
        summary[split] = {
            "images": len(rows),
            "boxes": {label: boxes[label] for label in CLASSES},
            "sessions": len({row["session"] for row in rows}),
        }
        for row in rows:
            session = row["session"]
            if session in owner and owner[session] != split:
                leaks.append((session, owner[session], split))
            owner[session] = split
    if leaks:
        raise AssertionError(f"phát hiện session leakage: {leaks[:5]}")
    return summary


def draw_ground_truth(
    entry: dict[str, Any], data_dir: str | Path = "data", line_width: int = 3
) -> Image.Image:
    """Vẽ ground-truth YOLO lên một bản sao PIL để hiển thị trong notebook."""
    with Image.open(image_path(entry, data_dir)) as source:
        canvas = source.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    colors = {"pothole": "#ff3b30", "crack": "#34c759", "manhole": "#007aff"}
    for obj in entry["objects"]:
        xc, yc, bw, bh = obj["yolo"]
        box = [
            int(round((xc - bw / 2) * canvas.width)),
            int(round((yc - bh / 2) * canvas.height)),
            int(round((xc + bw / 2) * canvas.width)),
            int(round((yc + bh / 2) * canvas.height)),
        ]
        color = colors[obj["label"]]
        draw.rectangle(box, outline=color, width=line_width)
        draw.text((box[0] + 2, max(0, box[1] - 12)), obj["label"], fill=color)
    return canvas


def sample_visualizations(
    split: str = "train",
    n: int = 6,
    *,
    seed: int = 42,
    data_dir: str | Path = "data",
) -> list[Image.Image]:
    rows = read_jsonl(Path(data_dir) / "splits" / f"{split}.jsonl")
    chosen = random.Random(seed).sample(rows, min(n, len(rows)))
    return [draw_ground_truth(row, data_dir) for row in chosen]


def draw_prediction_comparison(
    entry: dict[str, Any],
    raw_prediction: str,
    *,
    data_dir: str | Path = "data",
    coord: str = "norm1000",
    line_width: int = 3,
) -> Image.Image:
    """Vẽ GT nét xanh và prediction nét đỏ trên cùng một ảnh."""
    with Image.open(image_path(entry, data_dir)) as source:
        canvas = source.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for obj in entry["objects"]:
        xc, yc, bw, bh = obj["yolo"]
        box = [
            round((xc - bw / 2) * canvas.width),
            round((yc - bh / 2) * canvas.height),
            round((xc + bw / 2) * canvas.width),
            round((yc + bh / 2) * canvas.height),
        ]
        draw.rectangle(box, outline="#34c759", width=line_width)
        draw.text((box[0] + 2, max(0, box[1] - 12)), f"GT {obj['label']}", fill="#34c759")

    parsed = parse_output(raw_prediction)
    if parsed is None:
        draw.text((8, 8), "PRED: invalid JSON", fill="#ff3b30")
        return canvas
    for obj in parsed:
        x1, y1, x2, y2 = obj["bbox"]
        if coord == "norm1000":
            box = [
                round(x1 / 1000 * canvas.width),
                round(y1 / 1000 * canvas.height),
                round(x2 / 1000 * canvas.width),
                round(y2 / 1000 * canvas.height),
            ]
        elif coord == "pixel":
            box = [round(x1), round(y1), round(x2), round(y2)]
        else:
            raise ValueError("coord phải là 'norm1000' hoặc 'pixel'")
        draw.rectangle(box, outline="#ff3b30", width=line_width)
        draw.text((box[0] + 2, min(canvas.height - 12, box[1] + 2)),
                  f"P {obj['label']}", fill="#ff3b30")
    return canvas


def export_prediction_visualizations(
    pred_path: str | Path,
    output_dir: str | Path,
    *,
    split: str = "test",
    n: int = 6,
    seed: int = 42,
    data_dir: str | Path = "data",
    coord: str = "norm1000",
) -> list[Path]:
    """Xuất ảnh minh hoạ có thể commit vào ``report/figures``."""
    entries = read_jsonl(Path(data_dir) / "splits" / f"{split}.jsonl")
    predictions = {row["image_id"]: row["raw"] for row in read_jsonl(pred_path)}
    available = [entry for entry in entries if entry["image_id"] in predictions]
    if len(available) < n:
        raise ValueError(f"chỉ có {len(available)} prediction, không đủ xuất {n} ảnh")
    chosen = random.Random(seed).sample(available, n)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, entry in enumerate(chosen, 1):
        image = draw_prediction_comparison(
            entry, predictions[entry["image_id"]], data_dir=data_dir, coord=coord
        )
        path = output_dir / f"{index:02d}_{entry['image_id']}.jpg"
        image.save(path, quality=92)
        paths.append(path)
    return paths


def evidence_summary(evidence_dir: str | Path = "report/evidence") -> list[dict[str, Any]]:
    """Rút các cột cần điền vào bảng REPORT.md từ evidence thật."""
    rows = []
    for path in sorted(Path(evidence_dir).glob("e*.json")):
        evidence = json.loads(path.read_text(encoding="utf-8"))
        micro = evidence.get("micro", {})
        per_class = evidence.get("per_class", {})
        meta = evidence.get("meta", {})
        rows.append(
            {
                "name": evidence.get("name", path.stem),
                "trainable_params": meta.get("trainable_params"),
                "peak_vram_gb": meta.get("peak_vram_gb"),
                "train_minutes": meta.get("train_minutes"),
                "f1": micro.get("f1"),
                "ci95": evidence.get("f1_ci95"),
                "fp_per_image": evidence.get("fp_per_image"),
                **{
                    f"f1_{label}": per_class.get(label, {}).get("f1")
                    for label in CLASSES
                },
            }
        )
    return rows


def load_model_and_processor(model_id: str = MODEL_ID):
    """Nạp Qwen3-VL 4-bit; giữ ``visual`` và ``lm_head`` ngoài lượng tử hoá."""
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("Cần CUDA; hãy bật GPU accelerator cho notebook")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["visual", "lm_head"],
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization,
        dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return model, processor


def _user_message(image: Image.Image, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def encode_example(
    image: Image.Image,
    prompt: str,
    answer: str,
    processor: Any,
) -> dict[str, Any]:
    """Áp chat template và mask mọi token trước câu trả lời assistant."""
    user = _user_message(image, prompt)
    full = user + [{"role": "assistant", "content": answer}]
    prompt_enc = processor.apply_chat_template(
        user,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    full_enc = processor.apply_chat_template(
        full,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )
    n_prompt = prompt_enc["input_ids"].shape[1]
    if not (full_enc["input_ids"][0, :n_prompt] == prompt_enc["input_ids"][0]).all():
        raise AssertionError("chat template đầy đủ không có prompt làm tiền tố")
    labels = full_enc["input_ids"].clone()
    labels[0, :n_prompt] = -100

    result = {
        "input_ids": full_enc["input_ids"][0],
        "attention_mask": full_enc["attention_mask"][0],
        "labels": labels[0],
        "pixel_values": full_enc["pixel_values"],
        "image_grid_thw": full_enc["image_grid_thw"],
    }
    # Transformers 5.x có thể trả trường này cho Qwen3-VL.
    for key in ("mm_token_type_ids", "token_type_ids"):
        if key in full_enc:
            value = full_enc[key]
            result[key] = value[0] if value.ndim > 1 else value
    return result


def collate_fn(batch: Sequence[dict[str, Any]], pad_token_id: int):
    """Pad chuỗi văn bản, nhưng nối tensor ảnh dọc chiều 0 (không pad ảnh)."""
    import torch

    if not batch:
        raise ValueError("batch rỗng")
    padded = {
        "input_ids": torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"] for item in batch], batch_first=True, padding_value=pad_token_id
        ),
        "attention_mask": torch.nn.utils.rnn.pad_sequence(
            [item["attention_mask"] for item in batch], batch_first=True, padding_value=0
        ),
        "labels": torch.nn.utils.rnn.pad_sequence(
            [item["labels"] for item in batch], batch_first=True, padding_value=-100
        ),
        "pixel_values": torch.cat([item["pixel_values"] for item in batch], dim=0),
        "image_grid_thw": torch.cat([item["image_grid_thw"] for item in batch], dim=0),
    }
    for key in ("mm_token_type_ids", "token_type_ids"):
        if all(key in item for item in batch):
            padded[key] = torch.nn.utils.rnn.pad_sequence(
                [item[key] for item in batch], batch_first=True, padding_value=0
            )
    return padded


def vision_token_count(encoded: dict[str, Any], merge_size: int = 2) -> int:
    """Tính token ảnh sau spatial merge từ ``image_grid_thw``."""
    grid = encoded["image_grid_thw"]
    return int(sum(int(row.prod().item()) for row in grid) // (merge_size**2))


class RoadDamageDataset:
    """Torch-style Dataset mã hoá ảnh theo yêu cầu để không giữ toàn bộ pixel trong RAM."""

    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        processor: Any,
        *,
        data_dir: str | Path = "data",
        coord: str = "norm1000",
    ) -> None:
        self.rows = list(rows)
        self.processor = processor
        self.data_dir = Path(data_dir)
        self.coord = coord
        self.prompt = prompt_for_coord(coord)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        with Image.open(image_path(row, self.data_dir)) as source:
            image = source.convert("RGB")
        answer = objects_to_answer(
            row["objects"], coord=self.coord, width=row["width"], height=row["height"]
        )
        return encode_example(image, self.prompt, answer, self.processor)


def select_train_rows(
    rows: Sequence[dict[str, Any]], fraction: float = 1.0, seed: int = 42
) -> list[dict[str, Any]]:
    """Chọn subset E4 tất định; 10% của 1.422 ảnh cho đúng 142 ảnh."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction phải thuộc (0, 1]")
    rows = list(rows)
    if fraction == 1:
        return rows
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    n = max(1, int(len(rows) * fraction))
    return [rows[i] for i in sorted(indices[:n])]


def _target_pattern(setup: str) -> str:
    setup = setup.upper()
    if setup in {"E1", "E2"}:
        return _LLM_TARGET
    if setup == "E3":
        return rf"(?:{_LLM_TARGET[:-1]}|{_VIT_TARGET[:-1]})$"
    raise ValueError("setup phải là E1, E2 hoặc E3")


def apply_setup(
    model: Any,
    setup: str,
    *,
    rank: int = 16,
    alpha: int | None = None,
    dropout: float = 0.05,
    verify_counts: bool = True,
):
    """Gắn LoRA đúng module và mở projector float32 cho E2/E3."""
    import torch
    from peft import LoraConfig, get_peft_model

    setup = setup.upper()
    if rank <= 0:
        raise ValueError("rank phải dương")
    alpha = 2 * rank if alpha is None else alpha
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.enable_input_require_grads()

    pattern = _target_pattern(setup)
    matched = [name for name, _ in model.named_modules() if re.fullmatch(pattern, name)]
    expected = 348 if setup == "E3" else 252
    if verify_counts and len(matched) != expected:
        raise RuntimeError(
            f"{setup}: khớp {len(matched)} module LoRA, cần {expected}. "
            "Hãy kiểm tra model revision và in model.named_modules()."
        )
    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=pattern,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )

    if setup in {"E2", "E3"}:
        opened = []
        for name, parameter in peft_model.named_parameters():
            if "visual.merger." in name or "visual.deepstack_merger_list." in name:
                parameter.data = parameter.data.to(torch.float32)
                parameter.requires_grad = True
                opened.append(name)
        if not opened:
            raise RuntimeError("không tìm thấy visual.merger/deepstack_merger_list")

    # GradScaler không cho phép unscale gradient FP16. Tuỳ phiên bản
    # transformers/peft, adapter LoRA có thể kế thừa dtype FP16 của module gốc.
    # Giữ mọi tham số đang train ở FP32; trọng số base 4-bit/frozen không đổi.
    converted = []
    for name, parameter in peft_model.named_parameters():
        if parameter.requires_grad and parameter.dtype != torch.float32:
            parameter.data = parameter.data.to(torch.float32)
            converted.append(name)
    if converted:
        print(f"{setup}: converted {len(converted)} trainable tensors to float32")

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    print(
        f"{setup}: LoRA modules={len(matched)}, trainable={trainable:,} / "
        f"{total:,} ({100 * trainable / total:.3f}%)"
    )
    return peft_model


def _model_device(model: Any):
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("không xác định được device của model")


def _to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def projector_state_dict(model: Any) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if "visual.merger." in name or "visual.deepstack_merger_list." in name
    }


def save_training_artifacts(
    model: Any,
    output_dir: str | Path,
    *,
    save_projector: bool,
    run_meta: dict[str, Any] | None = None,
) -> None:
    """Lưu adapter và, khi cần, projector ngoài PEFT."""
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir / "adapter")
    if save_projector:
        state = projector_state_dict(model)
        if not state:
            raise RuntimeError("projector state rỗng")
        torch.save(state, output_dir / "projector.pt")
    if run_meta is not None:
        (output_dir / "run.json").write_text(
            json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def train_sft(
    model: Any,
    processor: Any,
    setup: str,
    *,
    rows: Sequence[dict[str, Any]] | None = None,
    data_dir: str | Path = "data",
    output_dir: str | Path | None = None,
    coord: str = "norm1000",
    batch_size: int = 2,
    grad_accum: int = 4,
    max_steps: int = 300,
    learning_rate: float = 2e-4,
    seed: int = 42,
    save_steps: int = 100,
) -> dict[str, Any]:
    """Vòng SFT fp16 với AdamW, warmup 5%, cosine và gradient clipping."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup

    setup = setup.upper()
    output_dir = Path(output_dir or f"work/{setup.lower()}")
    rows = list(rows) if rows is not None else read_jsonl(Path(data_dir) / "splits/train.jsonl")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dataset = RoadDamageDataset(rows, processor, data_dir=data_dir, coord=coord)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda items: collate_fn(items, processor.tokenizer.pad_token_id),
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    non_fp32 = [
        (name, str(parameter.dtype))
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.dtype != torch.float32
    ]
    if non_fp32:
        raise TypeError(
            "Mọi tham số trainable phải là float32 trước GradScaler; "
            f"phát hiện: {non_fp32[:5]}"
        )
    n_trainable = sum(parameter.numel() for parameter in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0)
    warmup_steps = max(1, round(0.05 * max_steps))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, max_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    device = _model_device(model)
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    iterator = iter(loader)
    loss_history: list[float] = []

    for step in range(1, max_steps + 1):
        accumulated = 0.0
        for _ in range(grad_accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = _to_device(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(**batch).loss
                scaled_loss = loss / grad_accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"loss không hữu hạn tại step {step}: {loss.item()}")
            scaler.scale(scaled_loss).backward()
            accumulated += float(loss.detach())

        # Một số kernel projector của Qwen3-VL trả gradient FP16 ngay cả khi
        # master parameter đã ở FP32. GradScaler từ chối unscale loại gradient
        # này. Chuyển gradient đã tích luỹ sang FP32 trước khi unscale để vẫn
        # giữ loss scaling, clipping và AdamW ở độ chính xác ổn định.
        promoted_grads = 0
        for parameter in trainable:
            if parameter.dtype != torch.float32:
                parameter.data = parameter.data.to(torch.float32)
            if parameter.grad is not None and parameter.grad.dtype != torch.float32:
                parameter.grad.data = parameter.grad.data.to(torch.float32)
                promoted_grads += 1
        if promoted_grads and (step == 1 or step % 10 == 0):
            print(f"{setup} step {step:03d}: promoted {promoted_grads} gradients to float32")
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        mean_loss = accumulated / grad_accum
        loss_history.append(mean_loss)

        if step == 1 or step % 10 == 0:
            print(
                f"{setup} step {step:03d}/{max_steps} loss={mean_loss:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )
        if save_steps and step % save_steps == 0 and step < max_steps:
            save_training_artifacts(
                model,
                output_dir / f"checkpoint-{step}",
                save_projector=setup in {"E2", "E3"},
            )

    minutes = (time.perf_counter() - started) / 60
    meta = {
        "setup": setup,
        "trainable_params": n_trainable,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "train_minutes": minutes,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "effective_batch_size": batch_size * grad_accum,
        "learning_rate": learning_rate,
        "coord": coord,
        "n_train_images": len(rows),
        "loss_first": loss_history[0],
        "loss_last": loss_history[-1],
    }
    save_training_artifacts(
        model, output_dir, save_projector=setup in {"E2", "E3"}, run_meta=meta
    )
    return meta


def train_experiment(
    setup: str,
    *,
    rank: int = 16,
    train_fraction: float = 1.0,
    coord: str = "norm1000",
    output_dir: str | Path | None = None,
    data_dir: str | Path = "data",
    max_steps: int = 300,
):
    """Tiện ích nạp model mới, áp setup rồi chạy một thí nghiệm SFT."""
    setup = setup.upper()
    model, processor = load_model_and_processor()
    model = apply_setup(model, setup, rank=rank)
    rows = select_train_rows(
        read_jsonl(Path(data_dir) / "splits/train.jsonl"), train_fraction, seed=42
    )
    meta = train_sft(
        model,
        processor,
        setup,
        rows=rows,
        data_dir=data_dir,
        output_dir=output_dir or f"work/{setup.lower()}",
        coord=coord,
        max_steps=max_steps,
    )
    return model, processor, meta


def load_trained_model(
    artifact_dir: str | Path,
    *,
    setup: str,
    model_id: str = MODEL_ID,
    is_trainable: bool = False,
):
    """Nạp base + adapter + projector; bật ``is_trainable`` khi tiếp tục E5."""
    import torch
    from peft import PeftModel

    artifact_dir = Path(artifact_dir)
    base, processor = load_model_and_processor(model_id)
    model = PeftModel.from_pretrained(
        base, artifact_dir / "adapter", is_trainable=is_trainable
    )
    projector_path = artifact_dir / "projector.pt"
    if setup.upper() in {"E2", "E3"}:
        if not projector_path.exists():
            raise FileNotFoundError(
                f"thiếu {projector_path}; suy luận lúc này sẽ làm mất projector đã train"
            )
        state = torch.load(projector_path, map_location="cpu", weights_only=True)
        incompatible = model.load_state_dict(state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        if unexpected:
            raise RuntimeError(f"projector có key không khớp: {unexpected[:5]}")
    if is_trainable:
        model.train()
        model.config.use_cache = False
    else:
        model.eval()
        model.config.use_cache = True
    return model, processor


def run_inference(
    model: Any,
    processor: Any,
    *,
    split: str = "test",
    output_path: str | Path = "work/predictions.jsonl",
    data_dir: str | Path = "data",
    coord: str = "norm1000",
    batch_size: int = 4,
    max_new_tokens: int = 256,
    resume: bool = True,
) -> list[dict[str, str]]:
    """Sinh JSONL; hỗ trợ tiếp tục từ file dở nếu notebook bị ngắt."""
    import torch

    rows = read_jsonl(Path(data_dir) / "splits" / f"{split}.jsonl")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, str]] = []
    done: set[str] = set()
    if resume and output_path.exists():
        predictions = read_jsonl(output_path)
        done = {row["image_id"] for row in predictions}
        if len(done) != len(predictions):
            raise ValueError(f"{output_path} có image_id trùng")
    else:
        output_path.write_text("", encoding="utf-8")

    remaining = [row for row in rows if row["image_id"] not in done]
    processor.tokenizer.padding_side = "left"
    device = _model_device(model)
    for start in range(0, len(remaining), batch_size):
        chunk = remaining[start : start + batch_size]
        images = []
        for row in chunk:
            with Image.open(image_path(row, data_dir)) as source:
                images.append(source.convert("RGB"))
        conversations = [_user_message(image, prompt_for_coord(coord)) for image in images]
        inputs = processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        inputs = _to_device(dict(inputs), device)
        prompt_length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        texts = processor.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        new_rows = [
            {"image_id": row["image_id"], "raw": text.strip()}
            for row, text in zip(chunk, texts, strict=True)
        ]
        with output_path.open("a", encoding="utf-8") as handle:
            for prediction in new_rows:
                handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        predictions.extend(new_rows)
        print(f"inference {min(start + len(chunk), len(remaining))}/{len(remaining)}")
    return predictions


def score_experiment(
    pred_path: str | Path,
    name: str,
    out_path: str | Path,
    *,
    split: str = "test",
    coord: str = "norm1000",
    run_meta_path: str | Path | None = None,
) -> None:
    """Gọi nguyên trạng ``scripts/score.py`` và tự lấy ba metadata từ run.json."""
    command = [
        sys.executable,
        "scripts/score.py",
        "--pred",
        str(pred_path),
        "--name",
        name,
        "--out",
        str(out_path),
        "--split",
        split,
        "--coord",
        coord,
    ]
    if run_meta_path:
        meta = json.loads(Path(run_meta_path).read_text(encoding="utf-8"))
        for key in ("trainable_params", "peak_vram_gb", "train_minutes"):
            command += ["--meta", f"{key}={meta[key]}"]
    subprocess.run(command, cwd=ROOT, check=True)


def _ground_truth_boxes(objects: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    boxes = []
    for obj in objects:
        converted = yolo_to_box(obj["label"], obj["yolo"])
        boxes.append({"label": converted["label"], "bbox": converted["bbox_2d"]})
    return boxes


def compute_reward(completion: str, gt_boxes: Sequence[dict[str, Any]]) -> float:
    """Verifiable IoU reward cho E5, trong [0, 1]."""
    pred = parse_output(completion)
    if pred is None:
        return 0.0
    if not gt_boxes and not pred:
        return 1.0

    normalized_gt = []
    for item in gt_boxes:
        bbox = item.get("bbox", item.get("bbox_2d"))
        if bbox is None:
            raise ValueError("GT box cần key 'bbox' hoặc 'bbox_2d'")
        normalized_gt.append({"label": item["label"], "bbox": tuple(bbox)})
    total_iou = 0.0
    labels = {item["label"] for item in normalized_gt} | {item["label"] for item in pred}
    for label in labels:
        gt = [item["bbox"] for item in normalized_gt if item["label"] == label]
        guessed = [item["bbox"] for item in pred if item["label"] == label]
        pairs = sorted(
            (
                (iou_xyxy(a, b), i, j)
                for i, a in enumerate(gt)
                for j, b in enumerate(guessed)
            ),
            reverse=True,
        )
        used_gt: set[int] = set()
        used_pred: set[int] = set()
        for value, i, j in pairs:
            if value < 0.5:
                break
            if i in used_gt or j in used_pred:
                continue
            used_gt.add(i)
            used_pred.add(j)
            total_iou += value
    denominator = max(len(normalized_gt), len(pred), 1)
    return 0.2 + 0.8 * total_iou / denominator


def check_reward_cases() -> dict[str, float]:
    """Năm ca bắt buộc của đề; ném AssertionError nếu sai."""
    gt = [{"label": "pothole", "bbox": [100, 100, 300, 300]}]
    perfect = '[{"label":"pothole","bbox_2d":[100,100,300,300]}]'
    cases = {
        "perfect": compute_reward(perfect, gt),
        "broken_json": compute_reward("not-json", gt),
        "wrong_label": compute_reward(
            '[{"label":"crack","bbox_2d":[100,100,300,300]}]', gt
        ),
        "extra_box": compute_reward(
            '[{"label":"pothole","bbox_2d":[100,100,300,300]},'
            '{"label":"pothole","bbox_2d":[500,500,600,600]}]',
            gt,
        ),
        "both_empty": compute_reward("[]", []),
    }
    expected = {
        "perfect": 1.0,
        "broken_json": 0.0,
        "wrong_label": 0.2,
        "extra_box": 0.6,
        "both_empty": 1.0,
    }
    for key, value in cases.items():
        if not math.isclose(value, expected[key], abs_tol=1e-9):
            raise AssertionError(f"reward {key}: {value} != {expected[key]}")
    return cases


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list) and completion:
        return _completion_text(completion[0])
    return ""


def grpo_iou_reward(
    completions: Sequence[Any], gt_boxes: Sequence[Sequence[dict[str, Any]]], **_: Any
) -> list[float]:
    """Adapter reward theo chữ ký custom reward của ``trl.GRPOTrainer``."""
    gt_boxes = list(gt_boxes)
    if len(gt_boxes) != len(completions):
        if not gt_boxes or len(completions) % len(gt_boxes):
            raise ValueError("không thể ghép completions với gt_boxes")
        repeat = len(completions) // len(gt_boxes)
        gt_boxes = [gt for gt in gt_boxes for _ in range(repeat)]
    return [
        compute_reward(_completion_text(completion), gt)
        for completion, gt in zip(completions, gt_boxes, strict=True)
    ]


def build_grpo_dataset(
    processor: Any,
    *,
    data_dir: str | Path = "data",
    split: str = "train",
    fraction: float = 1.0,
):
    """Tạo Dataset có ``prompt``, ``image`` và ``gt_boxes`` cho TRL."""
    from datasets import Dataset, Image as DatasetImage

    rows = select_train_rows(
        read_jsonl(Path(data_dir) / "splits" / f"{split}.jsonl"), fraction, seed=42
    )
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT_NORM1000},
            ],
        }
    ]
    rendered = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    dataset = Dataset.from_dict(
        {
            "prompt": [rendered] * len(rows),
            "image": [str(image_path(row, data_dir).resolve()) for row in rows],
            "gt_boxes": [_ground_truth_boxes(row["objects"]) for row in rows],
        }
    )
    return dataset.cast_column("image", DatasetImage())


def create_grpo_trainer(
    model: Any,
    processor: Any,
    train_dataset: Any,
    *,
    output_dir: str | Path = "work/e5",
):
    """Khởi tạo E5; gọi ``trainer.train()`` ở notebook khi đã kiểm tra reward."""
    from trl import GRPOConfig, GRPOTrainer

    processor.tokenizer.padding_side = "left"
    args = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=5e-6,
        max_steps=60,
        num_generations=4,
        max_completion_length=128,
        max_prompt_length=None,
        beta=0.0,
        use_vllm=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=True,
        remove_unused_columns=False,
        logging_steps=1,
        save_steps=20,
        report_to="none",
    )
    return GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=grpo_iou_reward,
        args=args,
        train_dataset=train_dataset,
    )


if __name__ == "__main__":
    print("Reward self-check:", check_reward_cases())
