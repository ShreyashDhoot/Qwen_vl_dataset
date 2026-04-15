#!/usr/bin/env python
"""Build and push an InternVL-labeled HF dataset from CSV + source dataset.

Default behavior:
- reads internvl_predictions.csv
- pulls rows from Aggshourya/auditor_cleaned (train split)
- matches CSV id to dataset index
- pushes merged rows to ShreyashDhoot/internvl-auditor

Output columns:
- index
- image
- prompt
- safe, nudity, violence, UNK
- label
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from datasets import Dataset, Features, Image as HFImage, Value, load_dataset, load_from_disk
from huggingface_hub import HfApi
from PIL import Image as PILImage

LABEL_COLUMNS = ["safe", "nudity", "violence", "UNK"]
PROMPT_CANDIDATES = ["prompt"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join InternVL CSV labels with source dataset rows and push to Hugging Face Hub."
    )
    parser.add_argument("--csv", default="internvl_predictions.csv", help="Path to InternVL predictions CSV.")
    parser.add_argument(
        "--source-dataset",
        default="Aggshourya/auditor_cleaned",
        help="HF dataset repo to read source rows from.",
    )
    parser.add_argument("--source-split", default="train", help="Source dataset split.")
    parser.add_argument(
        "--streaming",
        dest="streaming",
        action="store_true",
        default=True,
        help="Stream source dataset instead of materializing it on disk.",
    )
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Disable streaming and load source dataset fully into local cache.",
    )
    parser.add_argument(
        "--target-dataset",
        default="ShreyashDhoot/internvl-auditor",
        help="HF dataset repo to push merged rows to.",
    )
    parser.add_argument("--target-split", default="train", help="Target split name on hub.")
    parser.add_argument("--cache-dir", default="./hf_cache", help="HF cache directory.")
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Optional .env path to load HF token from.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional HF token (overrides env/.env).",
    )
    parser.add_argument(
        "--prompt-column",
        default=None,
        help="Optional explicit prompt column in source dataset. If omitted, common names are auto-detected.",
    )
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Optional custom commit message for each uploaded shard.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/update the target dataset as private.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5000,
        help="Print record-building progress every N scanned rows when streaming.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Rows per parquet shard when streaming and uploading.",
    )
    return parser.parse_args()


def load_dotenv_file(path: str | None) -> None:
    if not path:
        return

    env_path = Path(path)
    if not env_path.exists() or not env_path.is_file():
        return

    with env_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]

            if key and key not in os.environ:
                os.environ[key] = value


def resolve_hf_token(cli_token: str | None) -> str | None:
    if cli_token:
        return cli_token.strip() or None

    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value and value.strip():
            return value.strip()

    return None


def validate_hf_token(token: str) -> str:
    api = HfApi(token=token)
    try:
        user = api.whoami()
    except Exception as exc:
        raise RuntimeError(
            "Hugging Face authentication failed. Check token in --hf-token or .env (HF_TOKEN / HUGGINGFACE_HUB_TOKEN)."
        ) from exc

    if isinstance(user, dict):
        # huggingface_hub can return either {'name': ...} or {'auth': {'accessToken': ...}, ...}
        return str(user.get("name") or user.get("fullname") or "unknown")
    return str(user)


def detect_prompt_column(dataset, preferred: str | None) -> str:
    available = set(dataset.column_names)

    if preferred:
        if preferred not in available:
            raise ValueError(
                f"Prompt column '{preferred}' not found in source dataset columns: {sorted(available)}"
            )
        return preferred

    for name in PROMPT_CANDIDATES:
        if name in available:
            return name

    raise ValueError(
        "Could not infer prompt column. Pass --prompt-column explicitly. "
        f"Available columns: {sorted(available)}"
    )


def load_source_dataset(source_dataset: str, source_split: str, streaming: bool, cache_dir: str):
    source_path = Path(source_dataset)
    if source_path.exists():
        print(f"Loading local source dataset from disk: {source_path}")
        loaded = load_from_disk(str(source_path))
        if hasattr(loaded, "keys") and source_split in loaded:
            return loaded[source_split]
        return loaded

    return load_dataset(
        source_dataset,
        split=source_split,
        streaming=streaming,
        cache_dir=cache_dir,
    )


def read_csv_rows(csv_path: Path) -> dict[int, dict[str, int]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows_by_index: dict[int, dict[str, int]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "safe", "nudity", "violence"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        for row in reader:
            raw_id = str(row.get("id", "")).strip()
            if not raw_id:
                continue
            try:
                idx = int(raw_id)
            except ValueError:
                # Skip non-integer ids because this workflow matches by dataset index.
                continue

            label_values: dict[str, int] = {}
            for key in LABEL_COLUMNS:
                raw = str(row.get(key, "0")).strip()
                label_values[key] = int(float(raw)) if raw else 0

            rows_by_index[idx] = label_values

    return rows_by_index


def one_hot_label(values: dict[str, int]) -> str:
    for name in ["safe", "nudity", "violence", "UNK"]:
        if values.get(name, 0) == 1:
            return name
    return "UNK"


def build_record(item: dict[str, Any], idx: int, prompt_column: str, labels: dict[str, int]) -> dict[str, Any]:
    return {
        "index": idx,
        "image": normalize_image_for_hf(item["image"]),
        "prompt": item[prompt_column],
        "safe": int(labels.get("safe", 0)),
        "nudity": int(labels.get("nudity", 0)),
        "violence": int(labels.get("violence", 0)),
        "UNK": int(labels.get("UNK", 0)),
        "label": one_hot_label(labels),
    }


def upload_chunk(
    rows: list[dict[str, Any]],
    target_dataset: str,
    target_split: str,
    commit_message: str,
    token: str,
    part_number: int,
) -> None:
    if not rows:
        return

    ds = Dataset.from_list(rows, features=output_features())

    with tempfile.TemporaryDirectory(prefix="internvl_push_") as tmp_dir:
        ts = int(time.time())
        filename = f"{target_split}-append-{rows[0]['index']}-{rows[-1]['index']}-{ts}-part{part_number:04d}.parquet"
        local_path = Path(tmp_dir) / filename
        ds.to_parquet(str(local_path))

        print(f"Uploading chunk {part_number} with {len(rows):,} rows: {filename}")
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=f"data/{filename}",
            repo_id=target_dataset,
            repo_type="dataset",
            commit_message=commit_message,
            token=token,
        )


def build_and_upload_streaming(
    dataset,
    prompt_column: str,
    rows_by_index: dict[int, dict[str, int]],
    args: argparse.Namespace,
    hf_token: str,
) -> int:
    needed_indices = sorted(rows_by_index)
    if not needed_indices:
        return 0

    needed = set(needed_indices)
    max_needed = needed_indices[-1]
    found: set[int] = set()
    scanned = 0
    appended = 0
    part_number = 1
    batch: list[dict[str, Any]] = []

    print(f"Streaming scan up to source index {max_needed:,} for {len(needed):,} requested rows...")

    for idx, item in enumerate(dataset):
        scanned += 1
        if args.progress_every and scanned % args.progress_every == 0:
            print(f"Scanned {scanned:,} source rows, matched {len(found):,}/{len(needed):,}")

        if idx > max_needed and len(found) == len(needed):
            break

        if idx not in needed:
            continue

        labels = rows_by_index[idx]
        batch.append(build_record(item, idx, prompt_column, labels))
        found.add(idx)
        appended += 1

        if len(batch) >= args.chunk_size:
            commit_message = args.commit_message or (
                f"Append InternVL rows {batch[0]['index']}..{batch[-1]['index']} from {args.source_dataset}"
            )
            upload_chunk(
                rows=batch,
                target_dataset=args.target_dataset,
                target_split=args.target_split,
                commit_message=commit_message,
                token=hf_token,
                part_number=part_number,
            )
            batch = []
            part_number += 1

        if len(found) == len(needed):
            break

    if batch:
        commit_message = args.commit_message or (
            f"Append InternVL rows {batch[0]['index']}..{batch[-1]['index']} from {args.source_dataset}"
        )
        upload_chunk(
            rows=batch,
            target_dataset=args.target_dataset,
            target_split=args.target_split,
            commit_message=commit_message,
            token=hf_token,
            part_number=part_number,
        )

    missing_indices = sorted(needed - found)
    if missing_indices:
        print(
            f"Warning: {len(missing_indices)} requested indices were not found while streaming source split. "
            f"First few: {missing_indices[:10]}"
        )

    print(f"Scanned source rows: {scanned}")
    print(f"Final appended count: {appended}")
    return appended


def normalize_image_for_hf(image_obj):
    if isinstance(image_obj, PILImage.Image):
        return image_obj.convert("RGB")

    if isinstance(image_obj, dict):
        data = image_obj.get("bytes")
        if data:
            return PILImage.open(io.BytesIO(data)).convert("RGB")
        path = image_obj.get("path")
        if path:
            return PILImage.open(path).convert("RGB")

    # Some HF decoders expose image-like objects with .convert().
    if hasattr(image_obj, "convert"):
        return image_obj.convert("RGB")

    raise ValueError(f"Unsupported image format for upload: {type(image_obj)!r}")


def output_features() -> Features:
    return Features(
        {
            "index": Value("int64"),
            "image": HFImage(),
            "prompt": Value("string"),
            "safe": Value("int8"),
            "nudity": Value("int8"),
            "violence": Value("int8"),
            "UNK": Value("int8"),
            "label": Value("string"),
        }
    )


def build_records(dataset, prompt_column: str, rows_by_index: dict[int, dict[str, int]], progress_every: int) -> list[dict]:
    if hasattr(dataset, "__getitem__") and hasattr(dataset, "__len__"):
        return build_records_indexable(dataset, prompt_column, rows_by_index, progress_every)
    return build_records_streaming(dataset, prompt_column, rows_by_index, progress_every)


def build_records_indexable(
    dataset,
    prompt_column: str,
    rows_by_index: dict[int, dict[str, int]],
    progress_every: int,
) -> list[dict]:
    records: list[dict] = []
    dataset_len = len(dataset)

    missing_indices: list[int] = []
    for idx in sorted(rows_by_index):
        if progress_every > 0 and idx > 0 and idx % progress_every == 0:
            print(f"Building records: processed index {idx:,}/{dataset_len:,}")

        if idx < 0 or idx >= dataset_len:
            missing_indices.append(idx)
            continue

        item = dataset[idx]
        labels = rows_by_index[idx]

        record = {
            "index": idx,
            "image": normalize_image_for_hf(item["image"]),
            "prompt": item[prompt_column],
            "safe": int(labels.get("safe", 0)),
            "nudity": int(labels.get("nudity", 0)),
            "violence": int(labels.get("violence", 0)),
            "UNK": int(labels.get("UNK", 0)),
            "label": one_hot_label(labels),
        }

        records.append(record)

    if missing_indices:
        print(
            f"Warning: {len(missing_indices)} CSV indices were out of range for source split and were skipped. "
            f"First few: {missing_indices[:10]}"
        )

    return records


def build_records_streaming(
    dataset,
    prompt_column: str,
    rows_by_index: dict[int, dict[str, int]],
    progress_every: int,
) -> list[dict]:
    records: list[dict] = []
    needed_indices = sorted(rows_by_index)
    if not needed_indices:
        return records

    needed = set(needed_indices)
    max_needed = needed_indices[-1]
    found: set[int] = set()

    print(f"Streaming scan up to source index {max_needed:,} for {len(needed):,} requested rows...")

    for idx, item in enumerate(dataset):
        if progress_every > 0 and idx > 0 and idx % progress_every == 0:
            print(f"Scanned {idx:,} source rows, matched {len(found):,}/{len(needed):,}")

        if idx > max_needed and len(found) == len(needed):
            break

        if idx not in needed:
            continue

        labels = rows_by_index[idx]
        record = {
            "index": idx,
            "image": normalize_image_for_hf(item["image"]),
            "prompt": item[prompt_column],
            "safe": int(labels.get("safe", 0)),
            "nudity": int(labels.get("nudity", 0)),
            "violence": int(labels.get("violence", 0)),
            "UNK": int(labels.get("UNK", 0)),
            "label": one_hot_label(labels),
        }
        records.append(record)
        found.add(idx)

        if len(found) == len(needed):
            break

    missing_indices = sorted(needed - found)
    if missing_indices:
        print(
            f"Warning: {len(missing_indices)} CSV indices were not found while streaming source split and were skipped. "
            f"First few: {missing_indices[:10]}"
        )

    return records


def main() -> None:
    args = parse_args()
    load_dotenv_file(args.env_file)
    hf_token = resolve_hf_token(args.hf_token)
    if not hf_token:
        raise RuntimeError(
            "No HF token found. Put HF_TOKEN in .env, export HF_TOKEN, or pass --hf-token."
        )

    auth_user = validate_hf_token(hf_token)
    print(f"Authenticated to Hugging Face as: {auth_user}")

    csv_path = Path(args.csv)
    rows_by_index = read_csv_rows(csv_path)
    if not rows_by_index:
        raise RuntimeError("No valid rows found in CSV after parsing ids.")

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")

    print(f"Parsed CSV rows: {len(rows_by_index)}")
    print(f"Loading source dataset: {args.source_dataset} [{args.source_split}]")

    source_ds = load_source_dataset(args.source_dataset, args.source_split, args.streaming, args.cache_dir)

    prompt_column = detect_prompt_column(source_ds, args.prompt_column)
    print(f"Using prompt column: {prompt_column}")
    print("Building merged records... this can take time on large streaming splits.")

    if args.streaming:
        uploaded = build_and_upload_streaming(source_ds, prompt_column, rows_by_index, args, hf_token)
        print(f"Push complete. Rows uploaded: {uploaded}")
        return

    records = build_records(source_ds, prompt_column, rows_by_index, args.progress_every)
    if not records:
        raise RuntimeError("No merged records to push. Check CSV ids and source split.")

    out_ds = Dataset.from_list(records, features=output_features())
    print(f"Merged records ready: {len(out_ds)}")
    print(f"Pushing to hub: {args.target_dataset} [{args.target_split}]")

    out_ds.push_to_hub(
        args.target_dataset,
        split=args.target_split,
        private=args.private,
        token=hf_token,
    )

    print("Push complete.")


if __name__ == "__main__":
    main()
