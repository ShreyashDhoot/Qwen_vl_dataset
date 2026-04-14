#!/usr/bin/env python
"""Standalone InternVL annotation script converted from internvl.ipynb.

This version is safe to run over SSH:
- checks the pinned transformers version
- applies the torch.linspace monkey patch used in the notebook
- resumes from an existing CSV by skipping already-written rows
- appends to the same CSV instead of recreating it
- flushes progress to disk every few images
- writes UNK=1 when the model cannot confidently map a label
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import io
import os
import re
import sys
from itertools import islice
from pathlib import Path
from typing import Iterable

import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL_ID = "OpenGVLab/InternVL3_5-8B"
DEFAULT_DATASET = "Aggshourya/auditor_cleaned"
DEFAULT_SPLIT = "train"
DEFAULT_OUTPUT_CSV = "internvl_predictions.csv"
DEFAULT_TRANSFORMERS_VERSION = "4.51.3"
DEFAULT_FLUSH_EVERY = 5
DEFAULT_IMAGE_SIZE = 448

CATEGORIES = ["safe", "nudity", "violence"]
CSV_HEADER = ["id", *CATEGORIES, "UNK"]
LEGACY_CSV_HEADER = ["id", *CATEGORIES]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SYSTEM_PROMPT = """
You are a vision safety classifier used for dataset annotation.

Your task is to classify input images into exactly one of the following categories:

1. nudity
2. violence
3. safe

Classification Rules:
- Choose "nudity" if the image contains explicit sexual content, exposed genitalia, pornographic content, or clearly sexualized nudity.
- Choose "violence" if the image contains physical harm, gore, bloodshed, weapons in active violent use, assault, or graphic injury.
- Choose "safe" if neither nudity nor violence is clearly present.

Output Rules:
- Return EXACTLY one word.
- Your response MUST be one of: nudity, violence, safe.
- Do NOT explain your reasoning.
- Do NOT provide extra text, punctuation, formatting, or sentences.
- Do NOT identify yourself.
- If uncertain, choose the closest matching category conservatively.

You must follow these instructions strictly.
""".strip()

QUESTION = "<image>\nClassify this image into EXACTLY ONE of:\nnudity\nviolence\nsafe\n\nReturn ONLY one word."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the InternVL annotation pipeline with CSV resume support.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset name or local path.")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Dataset split to read.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="InternVL model identifier.")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV, help="Output CSV path.")
    parser.add_argument("--cache-dir", default="./hf_cache", help="HF cache directory.")
    parser.add_argument(
        "--streaming",
        dest="streaming",
        action="store_true",
        default=True,
        help="Stream the dataset instead of materializing it locally.",
    )
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Disable dataset streaming.",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume from an existing CSV by skipping already-written rows.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Start from the beginning even if the output CSV already exists.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=DEFAULT_FLUSH_EVERY,
        help="Flush and fsync the CSV every N newly written rows.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Resize images to this square size before inference.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on the number of new samples to process.",
    )
    parser.add_argument(
        "--transformers-version",
        default=DEFAULT_TRANSFORMERS_VERSION,
        help="Required transformers version pinned by the notebook.",
    )
    parser.add_argument(
        "--skip-transformers-version-check",
        action="store_true",
        help="Skip the transformers version guard.",
    )
    return parser.parse_args()


def ensure_transformers_version(required_version: str, skip_check: bool) -> None:
    if skip_check:
        return

    try:
        installed_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "transformers is not installed. Install it with: pip install -U transformers=="
            f"{required_version}"
        ) from exc

    if installed_version != required_version:
        raise RuntimeError(
            "This notebook was pinned to transformers=="
            f"{required_version}, but the installed version is {installed_version}.\n"
            f"Install the pinned version with: pip install -U transformers=={required_version}"
        )


def patch_torch_linspace() -> None:
    if hasattr(torch, "_real_linspace"):
        return

    torch._real_linspace = torch.linspace  # type: ignore[attr-defined]

    def _safe_linspace(*args, **kwargs):
        kwargs.setdefault("device", "cpu")
        return torch._real_linspace(*args, **kwargs)  # type: ignore[attr-defined]

    torch.linspace = _safe_linspace  # type: ignore[assignment]


def load_image(item: dict) -> Image.Image:
    image_obj = item["image"]
    if isinstance(image_obj, dict) and "bytes" in image_obj:
        return Image.open(io.BytesIO(image_obj["bytes"])).convert("RGB")
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")
    return image_obj.convert("RGB")


def preprocess_image(image: Image.Image, image_size: int) -> torch.Tensor:
    resample = getattr(Image, "Resampling", Image).BICUBIC
    resized = image.convert("RGB").resize((image_size, image_size), resample)
    width, height = resized.size

    pixel_buffer = torch.ByteTensor(torch.ByteStorage.from_buffer(resized.tobytes()))
    tensor = pixel_buffer.view(height, width, 3).permute(2, 0, 1).contiguous().float().div_(255.0)

    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def get_model_device(model) -> torch.device:
    model_device = getattr(model, "device", None)
    if isinstance(model_device, torch.device):
        return model_device
    if isinstance(model_device, str):
        return torch.device(model_device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer(model_id: str, cache_dir: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this model. Run on a GPU host over SSH.")

    print(f"Loading model: {model_id}")
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        use_flash_attn=False,
        trust_remote_code=True,
        device_map="auto",
        cache_dir=cache_dir,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=False,
        cache_dir=cache_dir,
    )

    model.system_message = SYSTEM_PROMPT
    return model, tokenizer


def extract_prediction(text: str) -> str | None:
    normalized = text.strip().lower()

    match = re.search(r"answer\s*:\s*(safe|nudity|violence)\b", normalized)
    if match:
        return match.group(1)

    for category in CATEGORIES:
        if re.search(rf"\b{re.escape(category)}\b", normalized):
            return category

    return None


def build_row(sample_id: str, prediction: str | None) -> dict[str, int | str]:
    row: dict[str, int | str] = {"id": sample_id, "UNK": 0}
    for category in CATEGORIES:
        row[category] = 0

    if prediction in CATEGORIES:
        row[prediction] = 1
    else:
        row["UNK"] = 1

    return row


def upgrade_legacy_csv(csv_path: Path) -> None:
    temp_path = csv_path.with_name(csv_path.name + ".tmp")

    with csv_path.open("r", encoding="utf-8", newline="") as source, temp_path.open(
        "w", encoding="utf-8", newline=""
    ) as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=CSV_HEADER)
        writer.writeheader()
        for row in reader:
            upgraded = {"id": row.get("id", ""), "UNK": 0}
            for category in CATEGORIES:
                upgraded[category] = int(str(row.get(category, "0")).strip() or 0)
            writer.writerow(upgraded)

    temp_path.replace(csv_path)


def count_existing_rows(csv_path: Path) -> int:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return 0

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            return 0
        if header == LEGACY_CSV_HEADER:
            upgrade_legacy_csv(csv_path)
            return count_existing_rows(csv_path)
        if header != CSV_HEADER:
            raise ValueError(
                f"Existing CSV header does not match the expected schema.\n"
                f"Found:   {header}\nExpected: {CSV_HEADER}\n"
                "Move or rename the old file before running this script."
            )
        return sum(1 for _ in reader)


def iter_dataset_from(dataset, start_index: int) -> Iterable[dict]:
    if start_index <= 0:
        return iter(dataset)

    skip = getattr(dataset, "skip", None)
    if callable(skip):
        return iter(skip(start_index))

    return islice(iter(dataset), start_index, None)


def flush_writer(handle) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def main() -> None:
    args = parse_args()
    ensure_transformers_version(args.transformers_version, args.skip_transformers_version_check)
    patch_torch_linspace()

    dataset = load_dataset(
        args.dataset,
        split=args.split,
        streaming=args.streaming,
        cache_dir=args.cache_dir,
    )
    model, tokenizer = load_model_and_tokenizer(args.model_id, args.cache_dir)
    device = get_model_device(model)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        existing_rows = count_existing_rows(output_path)
    else:
        existing_rows = 0
        if output_path.exists() and output_path.stat().st_size > 0:
            raise RuntimeError(
                f"Output CSV already exists at {output_path}. Use --resume to append or choose a new path."
            )

    print(f"Output CSV: {output_path}")
    print(f"Existing rows: {existing_rows}")
    print(f"Resuming from dataset index: {existing_rows}")

    open_mode = "a" if args.resume and output_path.exists() and output_path.stat().st_size > 0 else "w"
    written_since_flush = 0
    processed_new = 0
    total_seen = 0

    with output_path.open(open_mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_HEADER)
        if open_mode == "w":
            writer.writeheader()

        for dataset_index, item in enumerate(iter_dataset_from(dataset, existing_rows), start=existing_rows):
            if args.max_samples is not None and processed_new >= args.max_samples:
                break

            total_seen = dataset_index + 1
            try:
                image = load_image(item)
                pixel_values = preprocess_image(image, args.image_size).unsqueeze(0).to(device=device, dtype=torch.bfloat16)

                with torch.inference_mode():
                    response = model.chat(
                        tokenizer,
                        pixel_values,
                        QUESTION,
                        dict(max_new_tokens=16, do_sample=False),
                    )

                prediction = extract_prediction(response if isinstance(response, str) else str(response))
            except Exception as exc:
                prediction = None
                print(f"[{dataset_index}] inference failed: {exc}", file=sys.stderr)

            sample_id = str(item.get("id", dataset_index))
            writer.writerow(build_row(sample_id, prediction))
            processed_new += 1
            written_since_flush += 1

            if written_since_flush >= max(1, args.flush_every):
                flush_writer(handle)
                written_since_flush = 0
                print(f"Flushed after {processed_new} new rows (last dataset index: {dataset_index})")

        if written_since_flush > 0:
            flush_writer(handle)

    print(f"Done. New rows written: {processed_new}")
    print(f"Last dataset index processed: {total_seen - 1 if processed_new else existing_rows - 1}")


if __name__ == "__main__":
    main()
