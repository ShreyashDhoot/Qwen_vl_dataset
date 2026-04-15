#!/usr/bin/env python
"""Append NSFW samples from zxbsmk/NSFW-T2I into a target HF dataset.

Behavior:
- Reads source dataset zxbsmk/NSFW-T2I
- Filters rows where JSON field NSFW == "NSFW"
- Uses source fields:
  - image from "jpg"
  - prompt from "text"
- Assigns indices starting at 80000
- Uploads exactly up to 9000 rows (default)
- Appends by uploading a new parquet file to the dataset repo (does not delete/overwrite existing parquet files)

Note:
- You must be authenticated with Hugging Face (huggingface-cli login or HF_TOKEN env var).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from datasets import Dataset, Features, Image as HFImage, Value, load_dataset
from huggingface_hub import HfApi
from PIL import Image as PILImage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append NSFW-T2I samples to HF dataset as a new parquet shard.")
    parser.add_argument("--source-dataset", default="zxbsmk/NSFW-T2I", help="Source dataset repo.")
    parser.add_argument("--source-split", default="train", help="Source split.")
    parser.add_argument("--target-dataset", default="ShreyashDhoot/internvl-auditor", help="Target dataset repo.")
    parser.add_argument("--target-split", default="train", help="Target split label for output shard naming.")
    parser.add_argument("--cache-dir", default="./hf_cache", help="HF cache directory.")
    parser.add_argument("--max-upload", type=int, default=9000, help="Maximum number of NSFW samples to append.")
    parser.add_argument("--chunk-size", type=int, default=500, help="Rows per parquet shard upload.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5000,
        help="Print scan progress every N source rows.",
    )
    parser.add_argument("--start-index", type=int, default=80000, help="Starting index for appended rows.")
    parser.add_argument("--env-file", default=".env", help="Optional .env path to load HF token from.")
    parser.add_argument("--hf-token", default=None, help="Optional HF token (overrides env/.env).")
    parser.add_argument(
        "--label-value",
        default="nudity",
        help="Value to store in label column for appended rows (default: nudity).",
    )
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Optional custom commit message for Hub upload.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and validate rows without uploading parquet to the Hub.",
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
        who = api.whoami()
    except Exception as exc:
        raise RuntimeError(
            "Hugging Face authentication failed. Check token in --hf-token or .env (HF_TOKEN / HUGGINGFACE_HUB_TOKEN)."
        ) from exc

    if isinstance(who, dict):
        return str(who.get("name") or who.get("fullname") or "unknown")
    return str(who)


def parse_json_field(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            loaded = json.loads(text)
            if isinstance(loaded, dict):
                return loaded
        except json.JSONDecodeError:
            return {}
    return {}


def extract_nsfw_flag(sample: dict[str, Any]) -> str:
    # Some datasets store metadata directly as keys, others nest inside a JSON field.
    direct = sample.get("NSFW", sample.get("nsfw", None))
    if direct is not None:
        return str(direct).strip().upper()

    meta = parse_json_field(sample.get("json", sample.get("metadata", None)))
    if meta:
        value = meta.get("NSFW", meta.get("nsfw", ""))
        return str(value).strip().upper()

    return ""


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


def collect_existing_indices(target_dataset: str, target_split: str, cache_dir: str) -> set[int]:
    existing: set[int] = set()
    try:
        ds = load_dataset(target_dataset, split=target_split, streaming=True, cache_dir=cache_dir)
    except Exception:
        return existing

    for row in ds:
        raw = row.get("index", None)
        if raw is None:
            continue
        try:
            existing.add(int(raw))
        except (TypeError, ValueError):
            continue

    return existing


def build_row(sample: dict[str, Any], index_value: int, label_value: str) -> dict[str, Any]:
    return {
        "index": index_value,
        "image": normalize_image_for_hf(sample["jpg"]),
        "prompt": str(sample.get("text", "")),
        "safe": 1 if label_value == "safe" else 0,
        "nudity": 1 if label_value == "nudity" else 0,
        "violence": 1 if label_value == "violence" else 0,
        "UNK": 1 if label_value == "UNK" else 0,
        "label": label_value,
    }


def upload_new_parquet(
    rows: list[dict[str, Any]],
    target_dataset: str,
    target_split: str,
    commit_message: str,
    dry_run: bool,
    token: str,
    part_number: int,
) -> None:
    if not rows:
        raise RuntimeError("No rows to append. Check source fields and NSFW filter.")

    ds = Dataset.from_list(rows, features=output_features())

    with tempfile.TemporaryDirectory(prefix="nsfw_append_") as tmp_dir:
        ts = int(time.time())
        min_idx = min(int(r["index"]) for r in rows)
        max_idx = max(int(r["index"]) for r in rows)
        filename = f"{target_split}-append-{min_idx}-{max_idx}-{ts}-part{part_number:04d}.parquet"
        local_path = Path(tmp_dir) / filename
        repo_path = f"data/{filename}"

        ds.to_parquet(str(local_path))
        print(f"Wrote parquet shard: {local_path}")
        print(f"Will upload to repo path: {repo_path}")

        if dry_run:
            print("Dry run enabled: skipping Hub upload.")
            return

        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=target_dataset,
            repo_type="dataset",
            commit_message=commit_message,
            token=token,
        )

        print("Upload complete.")


def main() -> None:
    args = parse_args()
    load_dotenv_file(args.env_file)
    hf_token = resolve_hf_token(args.hf_token)
    if not hf_token:
        raise RuntimeError("No HF token found. Put HF_TOKEN in .env, export HF_TOKEN, or pass --hf-token.")

    auth_user = validate_hf_token(hf_token)
    print(f"Authenticated to Hugging Face as: {auth_user}")

    if args.max_upload <= 0:
        raise ValueError("--max-upload must be > 0")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")
    if args.label_value not in {"safe", "nudity", "violence", "UNK"}:
        raise ValueError("--label-value must be one of: safe, nudity, violence, UNK")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be >= 0")

    print(f"Reading existing target split to preserve append behavior: {args.target_dataset} [{args.target_split}]")
    existing_indices = collect_existing_indices(args.target_dataset, args.target_split, args.cache_dir)
    print(f"Existing indices found: {len(existing_indices)}")

    source = load_dataset(
        args.source_dataset,
        split=args.source_split,
        streaming=True,
        cache_dir=args.cache_dir,
    )

    scanned = 0
    appended = 0
    part_number = 1
    next_index = args.start_index
    batch: list[dict[str, Any]] = []

    print(
        f"Scanning source dataset for NSFW rows... target={args.max_upload:,}, "
        f"chunk-size={args.chunk_size:,}, start-index={args.start_index:,}"
    )

    for sample in source:
        scanned += 1
        if args.progress_every and scanned % args.progress_every == 0:
            print(f"Scanned {scanned:,} source rows, matched {appended:,}/{args.max_upload:,}")

        if appended >= args.max_upload:
            break

        nsfw_flag = extract_nsfw_flag(sample)
        if nsfw_flag != "NSFW":
            continue

        if "jpg" not in sample or "text" not in sample:
            continue

        while next_index in existing_indices:
            next_index += 1

        row = build_row(sample, next_index, args.label_value)
        batch.append(row)
        existing_indices.add(next_index)
        next_index += 1
        appended += 1

        if len(batch) >= args.chunk_size:
            commit_message = args.commit_message or (
                f"Append NSFW rows {batch[0]['index']}..{batch[-1]['index']} from {args.source_dataset}"
            )
            print(f"Uploading chunk {part_number} with {len(batch):,} rows...")
            upload_new_parquet(
                rows=batch,
                target_dataset=args.target_dataset,
                target_split=args.target_split,
                commit_message=commit_message,
                dry_run=args.dry_run,
                token=hf_token,
                part_number=part_number,
            )
            print(f"Uploaded chunk {part_number}: {len(batch)} rows")
            part_number += 1
            batch = []

    if batch:
        commit_message = args.commit_message or (
            f"Append NSFW rows {batch[0]['index']}..{batch[-1]['index']} from {args.source_dataset}"
        )
        print(f"Uploading final chunk {part_number} with {len(batch):,} rows...")
        upload_new_parquet(
            rows=batch,
            target_dataset=args.target_dataset,
            target_split=args.target_split,
            commit_message=commit_message,
            dry_run=args.dry_run,
            token=hf_token,
            part_number=part_number,
        )
        print(f"Uploaded chunk {part_number}: {len(batch)} rows")

    print(f"Scanned source rows: {scanned}")
    print(f"Final appended count: {appended}")
    if appended > 0:
        print(f"Index range: {args.start_index}..{next_index - 1}")


if __name__ == "__main__":
    main()
