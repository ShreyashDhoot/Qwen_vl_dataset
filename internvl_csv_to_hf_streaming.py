#!/usr/bin/env python
"""Build a new HF dataset from InternVL CSV labels using streaming-only source reads.

What this script does:
- Reads InternVL label CSV (id, safe, nudity, violence, UNK)
- Streams source rows from Aggshourya/auditor_cleaned without full dataset download
- Matches CSV `id` to streamed source row index
- Uploads parquet shards to a target HF dataset

Output columns:
- index
- image (encoded image bytes, HF Image feature)
- prompt
- safe, nudity, violence, UNK
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

from datasets import Dataset, Features, Image as HFImage, Value, load_dataset
from huggingface_hub import HfApi
from PIL import Image as PILImage

LABEL_COLUMNS = ["safe", "nudity", "violence", "UNK"]
PROMPT_CANDIDATES = ["prompt", "text", "caption"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream source dataset, join with InternVL CSV labels, and upload to a new HF dataset."
    )
    parser.add_argument("--csv", default="internvl_predictions.csv", help="Path to InternVL predictions CSV.")
    parser.add_argument(
        "--source-dataset",
        default="Aggshourya/auditor_cleaned",
        help="HF source dataset repo to stream rows from.",
    )
    parser.add_argument("--source-split", default="train", help="Source dataset split.")
    parser.add_argument(
        "--target-dataset",
        required=True,
        help="Target HF dataset repo id to create/update, e.g. username/internvl-auditor-v2.",
    )
    parser.add_argument("--target-split", default="train", help="Target split label for uploaded shard names.")
    parser.add_argument("--cache-dir", default="./hf_cache", help="HF cache directory.")
    parser.add_argument("--chunk-size", type=int, default=250, help="Rows per uploaded parquet shard.")
    parser.add_argument("--progress-every", type=int, default=1000, help="Print progress every N scanned rows.")
    parser.add_argument(
        "--prompt-column",
        default=None,
        help="Optional explicit prompt column in source dataset. Auto-detected if omitted.",
    )
    parser.add_argument("--env-file", default=".env", help="Optional .env path for HF token loading.")
    parser.add_argument("--hf-token", default=None, help="Optional HF token (overrides env/.env).")
    parser.add_argument("--private", action="store_true", help="Create target dataset as private if new.")
    parser.add_argument("--commit-message", default=None, help="Optional custom commit message per shard upload.")
    parser.add_argument(
        "--skip-remote-verify",
        action="store_true",
        help="Skip remote verification of image feature after first uploaded shard.",
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
        return str(user.get("name") or user.get("fullname") or "unknown")
    return str(user)


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
                continue

            label_values: dict[str, int] = {}
            for key in LABEL_COLUMNS:
                raw = str(row.get(key, "0")).strip()
                label_values[key] = int(float(raw)) if raw else 0

            rows_by_index[idx] = label_values

    return rows_by_index


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
        }
    )


def detect_prompt_column(dataset, preferred: str | None) -> str:
    available = set(dataset.column_names)

    if preferred:
        if preferred not in available:
            raise ValueError(f"Prompt column '{preferred}' not found. Available columns: {sorted(available)}")
        return preferred

    for name in PROMPT_CANDIDATES:
        if name in available:
            return name

    raise ValueError(
        "Could not infer prompt column. Pass --prompt-column explicitly. "
        f"Available columns: {sorted(available)}"
    )


def pil_to_hf_image_dict(img: PILImage.Image) -> dict[str, Any]:
    rgb = img.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, format="PNG")
    return {"bytes": buf.getvalue(), "path": None}


def normalize_image_for_hf(image_obj: Any) -> dict[str, Any]:
    if isinstance(image_obj, PILImage.Image):
        return pil_to_hf_image_dict(image_obj)

    if isinstance(image_obj, dict):
        data = image_obj.get("bytes")
        if data:
            with PILImage.open(io.BytesIO(data)) as img:
                return pil_to_hf_image_dict(img)

        path = image_obj.get("path")
        if path:
            with PILImage.open(path) as img:
                return pil_to_hf_image_dict(img)

    if isinstance(image_obj, str) and image_obj:
        with PILImage.open(image_obj) as img:
            return pil_to_hf_image_dict(img)

    if hasattr(image_obj, "convert"):
        return pil_to_hf_image_dict(image_obj)

    raise ValueError(f"Unsupported image format for upload: {type(image_obj)!r}")


def build_record(item: dict[str, Any], idx: int, prompt_column: str, labels: dict[str, int]) -> dict[str, Any]:
    return {
        "index": idx,
        "image": normalize_image_for_hf(item["image"]),
        "prompt": str(item.get(prompt_column, "")),
        "safe": int(labels.get("safe", 0)),
        "nudity": int(labels.get("nudity", 0)),
        "violence": int(labels.get("violence", 0)),
        "UNK": int(labels.get("UNK", 0)),
    }


def upload_chunk(
    rows: list[dict[str, Any]],
    target_dataset: str,
    target_split: str,
    commit_message: str,
    token: str,
    part_number: int,
) -> None:
    ds = Dataset.from_list(rows, features=output_features())
    image_feature = ds.features.get("image")
    if not isinstance(image_feature, HFImage):
        raise RuntimeError("Refusing upload: local dataset image column is not HF Image feature.")

    with tempfile.TemporaryDirectory(prefix="internvl_stream_push_") as tmp_dir:
        ts = int(time.time())
        filename = f"{target_split}-internvl-{rows[0]['index']}-{rows[-1]['index']}-{ts}-part{part_number:04d}.parquet"
        local_path = Path(tmp_dir) / filename
        ds.to_parquet(str(local_path))

        api = HfApi(token=token)
        print(f"Uploading chunk {part_number} with {len(rows):,} rows: {filename}")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=f"data/{filename}",
            repo_id=target_dataset,
            repo_type="dataset",
            commit_message=commit_message,
            token=token,
        )


def verify_remote_image_feature(target_dataset: str, target_split: str, cache_dir: str, token: str) -> None:
    remote_ds = load_dataset(
        target_dataset,
        split=target_split,
        streaming=True,
        cache_dir=cache_dir,
        token=token,
    )

    remote_features = getattr(remote_ds, "features", None)
    if not remote_features or "image" not in remote_features:
        raise RuntimeError("Remote verification failed: image column is missing in uploaded dataset.")

    if not isinstance(remote_features["image"], HFImage):
        raise RuntimeError(
            "Remote verification failed: uploaded image column is not HF Image feature. "
            "Aborting to avoid producing wrong dataset format."
        )

    first = next(iter(remote_ds))
    image_value = first.get("image")
    if not isinstance(image_value, PILImage.Image):
        raise RuntimeError(
            "Remote verification failed: first row image is not decoded as an image object. "
            f"Got type {type(image_value)!r}."
        )

    print("Remote verification passed: image column is HF Image and decodes to an image object.")


def ensure_target_repo(target_dataset: str, token: str, private: bool) -> None:
    api = HfApi(token=token)
    api.create_repo(repo_id=target_dataset, repo_type="dataset", private=private, exist_ok=True, token=token)


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def stream_join_and_upload(dataset, prompt_column: str, rows_by_index: dict[int, dict[str, int]], args, token: str) -> int:
    needed = set(rows_by_index)
    if not needed:
        return 0

    max_needed = max(needed)
    found: set[int] = set()
    scanned = 0
    uploaded = 0
    part_number = 1
    batch: list[dict[str, Any]] = []
    start_time = time.time()
    verified_remote = False

    print(f"Streaming source up to index {max_needed:,} for {len(needed):,} requested ids (no full download).")

    for idx, item in enumerate(dataset):
        scanned += 1

        if args.progress_every > 0 and scanned % args.progress_every == 0:
            elapsed = time.time() - start_time
            rate = scanned / elapsed if elapsed > 0 else 0.0
            pct = (100.0 * min(scanned, max_needed + 1) / (max_needed + 1)) if max_needed >= 0 else 100.0
            print(
                "[stream] "
                f"scanned={scanned:,} ({pct:.1f}%) "
                f"matched={len(found):,}/{len(needed):,} "
                f"rate={rate:,.0f} rows/s "
                f"elapsed={format_duration(elapsed)}"
            )

        if idx > max_needed:
            break

        if idx not in needed:
            continue

        labels = rows_by_index[idx]
        batch.append(build_record(item, idx, prompt_column, labels))
        found.add(idx)
        uploaded += 1

        if len(batch) >= args.chunk_size:
            commit_message = args.commit_message or (
                f"Add InternVL rows {batch[0]['index']}..{batch[-1]['index']} from {args.source_dataset}"
            )
            upload_chunk(
                rows=batch,
                target_dataset=args.target_dataset,
                target_split=args.target_split,
                commit_message=commit_message,
                token=token,
                part_number=part_number,
            )
            if not args.skip_remote_verify and not verified_remote:
                verify_remote_image_feature(args.target_dataset, args.target_split, args.cache_dir, token)
                verified_remote = True
            batch = []
            part_number += 1

        if len(found) == len(needed):
            break

    if batch:
        commit_message = args.commit_message or (
            f"Add InternVL rows {batch[0]['index']}..{batch[-1]['index']} from {args.source_dataset}"
        )
        upload_chunk(
            rows=batch,
            target_dataset=args.target_dataset,
            target_split=args.target_split,
            commit_message=commit_message,
            token=token,
            part_number=part_number,
        )
        if not args.skip_remote_verify and not verified_remote:
            verify_remote_image_feature(args.target_dataset, args.target_split, args.cache_dir, token)

    missing = sorted(needed - found)
    if missing:
        print(
            f"Warning: {len(missing)} CSV ids were not found in streamed source scan. "
            f"First few: {missing[:10]}"
        )

    print(f"Scanned source rows: {scanned:,}")
    print(f"Uploaded rows: {uploaded:,}")
    return uploaded


def main() -> None:
    args = parse_args()

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")

    load_dotenv_file(args.env_file)
    token = resolve_hf_token(args.hf_token)
    if not token:
        raise RuntimeError("No HF token found. Set HF_TOKEN/HUGGINGFACE_HUB_TOKEN, .env, or --hf-token.")

    auth_user = validate_hf_token(token)
    print(f"Authenticated to Hugging Face as: {auth_user}")

    csv_path = Path(args.csv)
    rows_by_index = read_csv_rows(csv_path)
    if not rows_by_index:
        raise RuntimeError("No valid CSV rows found.")

    print(f"Parsed CSV rows: {len(rows_by_index):,}")
    print(f"Loading source dataset in streaming mode: {args.source_dataset} [{args.source_split}]")
    source_ds = load_dataset(
        args.source_dataset,
        split=args.source_split,
        streaming=True,
        cache_dir=args.cache_dir,
    )

    prompt_column = detect_prompt_column(source_ds, args.prompt_column)
    print(f"Using prompt column: {prompt_column}")

    ensure_target_repo(args.target_dataset, token, args.private)
    total = stream_join_and_upload(source_ds, prompt_column, rows_by_index, args, token)
    print(f"Done. Total rows uploaded: {total:,}")


if __name__ == "__main__":
    main()
