#!/usr/bin/env python
"""Standalone conversion of the qwen-vlm-trial notebook."""

from __future__ import annotations

import argparse
import csv
import io
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

CATEGORIES = ["safe", "nudity", "violence", "hate"]
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_DATASET = "Aggshourya/auditor_cleaned"
DEFAULT_SPLIT = "train"
DEFAULT_OUTPUT_CSV = "Qwen_dataset_clean.csv"

PROMPT = """
You are an image safety classifier.

Classify the image into ONE category: safe, nudity, violence, or hate.

Use the following strict rules:

SAFE:
- No explicit nudity, no violence, no hate symbols or gestures.

NUDITY:
1. Is it a living thing? (yes/no)
2. Is it a human being? (yes/no)
3. Are private parts visible? (yes/no)
4. Is the person clothed ? (yes/no)
4. are the private parts genetalia ? (yes/no)

VIOLENCE:
1) do you see blood ? (yes/no)
2) do you see weapons ? (yes/no)
3) are the weapons used in active state of violence ? (yes/no)
4) are there humans in the photo ? (yes/no)
5) are the humans fighting each other ? (yes.no)

HATE:
1) does the image show racial bias? (yes/no)
2) does the image show hate symbols ? (yes/no)
3) does the image show hate ? (yes/no)

IMPORTANT:
Your final answer MUST be consistent with your reasoning.
Do not contradict your reasoning.

Instructions:
1. Describe what you SEE in the image briefly.
2. Explain which rule applies.
3. Output ONLY in this format:

Reasoning: <your reasoning>
Answer: <one word: safe / nudity / violence / hate>
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Qwen VLM safety classification pipeline.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset name or path.")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Dataset split to stream.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Qwen model identifier.")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV, help="Output CSV path.")
    parser.add_argument(
        "--cache-dir",
        default="./hf_cache",
        help="Local cache directory for model and dataset downloads.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Enable HF streaming mode. Default is local cached dataset iteration for speed.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference batch size. Increase to use more VRAM and improve throughput.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=80,
        help="Maximum generated tokens per sample.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print throughput stats every N processed samples.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit for the number of streamed samples to process.",
    )
    return parser.parse_args()


def load_raw_image(item: dict) -> Image.Image:
    img_data = item["image"]
    if isinstance(img_data, dict):
        return Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
    return img_data


def load_model_and_processor(model_id: str, cache_dir: str):
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": "auto",
        "cache_dir": cache_dir,
    }
    if torch.cuda.is_available():
        model_kwargs["attn_implementation"] = "flash_attention_2"

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
    except Exception as exc:
        if "attn_implementation" in model_kwargs:
            print(f"flash_attention_2 unavailable ({exc}); falling back to default attention.")
            model_kwargs.pop("attn_implementation", None)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
        else:
            raise

    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    return model, processor


def extract_prediction(output_text: str) -> str:
    match = re.search(r"answer:\s*(safe|nudity|violence|hate)", output_text)
    return match.group(1) if match else "safe"


def build_messages(image: Image.Image) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]


def process_batch(
    batch_items: list[dict],
    model,
    processor,
    device: str,
    max_new_tokens: int,
) -> list[tuple[str, str]]:
    messages_batch: list[list[dict]] = []
    ids: list[str] = []

    for item in batch_items:
        raw_image = load_raw_image(item)
        img_id = str(item.get("id", "unknown"))
        ids.append(img_id)
        messages_batch.append(build_messages(raw_image))

    texts = [
        processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_batch
    ]

    image_inputs: list = []
    for messages in messages_batch:
        img_inputs, _ = process_vision_info(messages)
        image_inputs.extend(img_inputs)

    inputs = processor(
        text=texts,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
        else:
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

    output_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    results: list[tuple[str, str]] = []
    for img_id, output_text in zip(ids, output_texts):
        predicted = extract_prediction(output_text.strip().lower())
        results.append((img_id, predicted))
    return results


def main() -> None:
    args = parse_args()
    dataset = load_dataset(
        args.dataset,
        split=args.split,
        streaming=args.streaming,
        cache_dir=args.cache_dir,
    )
    model, processor = load_model_and_processor(args.model_id, args.cache_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = ["id"] + CATEGORIES

    with output_path.open(mode="w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()

        processed = 0
        failed = 0
        started_at = time.time()
        batch: list[dict] = []

        for index, item in enumerate(dataset, start=1):
            if args.max_samples is not None and index > args.max_samples:
                break

            batch.append(item)
            if len(batch) < args.batch_size:
                continue

            try:
                results = process_batch(
                    batch_items=batch,
                    model=model,
                    processor=processor,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                )
                for img_id, predicted in results:
                    row = {category: 0 for category in CATEGORIES}
                    row["id"] = img_id
                    if predicted in CATEGORIES:
                        row[predicted] = 1
                    writer.writerow(row)
                    processed += 1
            except Exception as exc:
                failed += len(batch)
                print(f"Error processing batch ending at sample {index}: {exc}")
            finally:
                batch = []

            if args.log_every > 0 and processed > 0 and processed % args.log_every == 0:
                elapsed = max(time.time() - started_at, 1e-6)
                speed = processed / elapsed
                print(
                    f"Processed={processed} Failed={failed} "
                    f"Throughput={speed:.2f} samples/sec"
                )

        if batch:
            try:
                results = process_batch(
                    batch_items=batch,
                    model=model,
                    processor=processor,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                )
                for img_id, predicted in results:
                    row = {category: 0 for category in CATEGORIES}
                    row["id"] = img_id
                    if predicted in CATEGORIES:
                        row[predicted] = 1
                    writer.writerow(row)
                    processed += 1
            except Exception as exc:
                failed += len(batch)
                print(f"Error processing final batch: {exc}")

        elapsed = max(time.time() - started_at, 1e-6)
        speed = processed / elapsed
        print(
            f"Completed. Processed={processed} Failed={failed} "
            f"Elapsed={elapsed:.1f}s Throughput={speed:.2f} samples/sec"
        )


if __name__ == "__main__":
    main()
