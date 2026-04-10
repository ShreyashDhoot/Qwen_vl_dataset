#!/usr/bin/env python
"""Standalone conversion of the qwen-vlm-trial notebook."""

from __future__ import annotations

import argparse
import csv
import io
import re
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


def load_model_and_processor(model_id: str):
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
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


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset, streaming=True, split=args.split)
    model, processor = load_model_and_processor(args.model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = ["id"] + CATEGORIES

    with output_path.open(mode="w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()

        for index, item in enumerate(dataset, start=1):
            if args.max_samples is not None and index > args.max_samples:
                break

            try:
                raw_image = load_raw_image(item)
                img_id = item.get("id", f"unknown_{index}")
                messages = build_messages(raw_image)

                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                image_inputs, _ = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(device)

                with torch.inference_mode():
                    generated_ids = model.generate(**inputs, max_new_tokens=100)

                output_text = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )[0].strip().lower()
                predicted = extract_prediction(output_text)

                row = {category: 0 for category in CATEGORIES}
                row["id"] = img_id
                if predicted in CATEGORIES:
                    row[predicted] = 1

                writer.writerow(row)
                print(f"Sample {index} | ID: {img_id}")
                print(f"Final Prediction: {predicted}")
            except Exception as exc:
                print(f"Error processing Sample {index} (ID: {item.get('id', 'N/A')}): {exc}")
                continue


if __name__ == "__main__":
    main()
