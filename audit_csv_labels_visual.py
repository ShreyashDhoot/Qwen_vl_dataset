#!/usr/bin/env python
"""Visual audit tool: match CSV ids to dataset images and render labels into a PDF.

This script is separate from the VLM inference pipeline and does not modify it.
"""

from __future__ import annotations

import argparse
import csv
import io
from math import ceil
from pathlib import Path

from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont

DEFAULT_CSV = "Qwen_dataset_clean.csv"
DEFAULT_DATASET = "Aggshourya/auditor_cleaned"
DEFAULT_SPLIT = "train"
DEFAULT_OUT = "visual_label_audit.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a visual PDF audit from CSV labels and dataset images.")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="CSV with id and one-hot label columns.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset name/path.")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Dataset split.")
    parser.add_argument("--cache-dir", default="./hf_cache", help="HF cache directory.")
    parser.add_argument("--output", default=DEFAULT_OUT, help="Output PDF filename.")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to render.")
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="If set, pick rows at fixed interval (0, stride, 2*stride, ...). Overrides --samples.",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=["interval", "recent"],
        default="interval",
        help="interval = evenly spread across CSV, recent = last N rows.",
    )
    parser.add_argument("--id-column", default="id", help="CSV column used to locate dataset sample.")
    parser.add_argument("--columns", type=int, default=2, help="Grid columns in output PDF.")
    parser.add_argument("--image-size", type=int, default=420, help="Rendered image size per tile.")
    return parser.parse_args()


def read_csv_rows(csv_path: Path) -> tuple[list[dict], list[str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if reader.fieldnames is None:
            raise ValueError("CSV header missing.")
        return rows, reader.fieldnames


def infer_label_columns(fieldnames: list[str], id_column: str) -> list[str]:
    return [name for name in fieldnames if name != id_column]


def one_hot_to_label(row: dict, label_columns: list[str]) -> str:
    for label in label_columns:
        value = str(row.get(label, "")).strip()
        if value in {"1", "1.0", "true", "True"}:
            return label
    return "unknown"


def select_rows(rows: list[dict], count: int, mode: str, stride: int | None = None) -> list[dict]:
    if not rows:
        return []

    if stride is not None:
        stride = max(1, stride)
        picked = rows[::stride]
        if not picked:
            return [rows[0]]
        return picked

    count = max(1, min(count, len(rows)))

    if mode == "recent":
        return rows[-count:]

    if count == 1:
        return [rows[-1]]

    max_idx = len(rows) - 1
    picked: list[dict] = []
    used = set()
    for i in range(count):
        idx = round((i * max_idx) / (count - 1))
        if idx in used:
            continue
        used.add(idx)
        picked.append(rows[idx])

    if len(picked) < count:
        for idx in range(len(rows)):
            if idx not in used:
                picked.append(rows[idx])
                if len(picked) == count:
                    break

    return picked


def to_rgb_image(image_obj) -> Image.Image:
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")
    if isinstance(image_obj, dict) and "bytes" in image_obj:
        return Image.open(io.BytesIO(image_obj["bytes"])).convert("RGB")
    raise ValueError("Unsupported image object format.")


def resolve_dataset_item(dataset, csv_id: str):
    csv_id = str(csv_id)

    # Fast path: CSV id is dataset index.
    try:
        idx = int(csv_id)
        if 0 <= idx < len(dataset):
            item = dataset[idx]
            item_id = str(item.get("id", idx))
            return item, item_id, idx
    except ValueError:
        pass

    # Fallback: linear scan by item["id"].
    for idx, item in enumerate(dataset):
        if str(item.get("id", "")) == csv_id:
            item_id = str(item.get("id", idx))
            return item, item_id, idx

    return None, "not-found", -1


def build_tile(
    image: Image.Image | None,
    header_text: str,
    label_text: str,
    width: int,
    image_size: int,
) -> Image.Image:
    header_h = 76
    label_h = 44
    margin = 10
    total_h = header_h + image_size + label_h + (margin * 2)

    tile = Image.new("RGB", (width, total_h), color=(248, 248, 248))
    draw = ImageDraw.Draw(tile)
    font = ImageFont.load_default()

    draw.rectangle([(0, 0), (width - 1, total_h - 1)], outline=(120, 120, 120), width=2)
    draw.text((12, 10), header_text, fill=(20, 20, 20), font=font)

    image_box_top = header_h
    if image is not None:
        img = image.copy()
        img.thumbnail((width - 2 * margin, image_size), Image.Resampling.LANCZOS)
        x = (width - img.width) // 2
        y = image_box_top + (image_size - img.height) // 2
        tile.paste(img, (x, y))
    else:
        draw.rectangle(
            [(margin, image_box_top), (width - margin, image_box_top + image_size)],
            outline=(180, 50, 50),
            width=2,
        )
        draw.text((margin + 10, image_box_top + 10), "Image not found", fill=(180, 50, 50), font=font)

    draw.text((12, image_box_top + image_size + 10), label_text, fill=(10, 60, 140), font=font)
    return tile


def render_pdf(tiles: list[Image.Image], output_path: Path, columns: int) -> None:
    if not tiles:
        raise ValueError("No tiles to render.")

    columns = max(1, columns)
    rows = ceil(len(tiles) / columns)

    tile_w, tile_h = tiles[0].size
    page = Image.new("RGB", (tile_w * columns, tile_h * rows), color=(255, 255, 255))

    for i, tile in enumerate(tiles):
        r = i // columns
        c = i % columns
        page.paste(tile, (c * tile_w, r * tile_h))

    page.save(output_path, "PDF", resolution=150.0)


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv)
    rows, fieldnames = read_csv_rows(csv_path)
    label_columns = infer_label_columns(fieldnames, args.id_column)

    if not label_columns:
        raise ValueError(f"No label columns found in CSV (id column: {args.id_column}).")

    chosen_rows = select_rows(rows, args.samples, args.sampling_mode, args.stride)

    print(f"Loaded CSV rows: {len(rows)}")
    mode_text = f"stride={args.stride}" if args.stride is not None else args.sampling_mode
    print(f"Selected rows: {len(chosen_rows)} ({mode_text})")
    print(f"Loading dataset: {args.dataset} [{args.split}]")

    dataset = load_dataset(args.dataset, split=args.split, cache_dir=args.cache_dir)

    tiles: list[Image.Image] = []
    for order, row in enumerate(chosen_rows, start=1):
        csv_id = str(row.get(args.id_column, ""))
        predicted = one_hot_to_label(row, label_columns)

        item, matched_id, ds_index = resolve_dataset_item(dataset, csv_id)

        if item is None:
            header = f"#{order} | csv_id={csv_id} | dataset=NOT_FOUND"
            label = f"predicted={predicted}"
            tile = build_tile(None, header, label, width=args.image_size + 20, image_size=args.image_size)
            tiles.append(tile)
            print(f"[{order}] csv_id={csv_id} -> dataset NOT FOUND | predicted={predicted}")
            continue

        try:
            image = to_rgb_image(item.get("image"))
        except Exception:
            image = None

        header = f"#{order} | csv_id={csv_id} | ds_idx={ds_index} | ds_id={matched_id}"
        label = f"predicted={predicted}"
        tile = build_tile(image, header, label, width=args.image_size + 20, image_size=args.image_size)
        tiles.append(tile)

        print(f"[{order}] csv_id={csv_id} -> ds_idx={ds_index}, ds_id={matched_id}, predicted={predicted}")

    out_path = Path(args.output)
    render_pdf(tiles, out_path, args.columns)
    print(f"Saved visual audit PDF: {out_path}")


if __name__ == "__main__":
    main()
