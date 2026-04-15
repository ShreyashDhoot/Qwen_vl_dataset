#!/usr/bin/env python
"""Live monitor for category counts in internvl_predictions.csv.

Continuously reads the CSV and prints running totals for one-hot label columns.
Designed to run alongside internvl.py while it appends rows.
"""

from __future__ import annotations

import argparse
import csv
import signal
import sys
import time
from pathlib import Path

DEFAULT_CSV = "internvl_predictions.csv"
DEFAULT_REFRESH = 2.0

DEFAULT_CATEGORIES = ["safe", "nudity", "violence", "UNK"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show live category counts from prediction CSV.")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="CSV file to monitor.")
    parser.add_argument("--refresh", type=float, default=DEFAULT_REFRESH, help="Refresh interval in seconds.")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        help="Category columns to count as one-hot labels.",
    )
    parser.add_argument(
        "--show-total-rows",
        action="store_true",
        help="Show raw data row count (excluding header) in addition to counted labels.",
    )
    return parser.parse_args()


def as_one_hot_int(value: str | None) -> int:
    if value is None:
        return 0
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "yes"}:
        return 1
    return 0


def read_counts(csv_path: Path, categories: list[str]) -> tuple[dict[str, int], int, int]:
    counts = {name: 0 for name in categories}
    valid_rows = 0
    total_rows = 0

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return counts, valid_rows, total_rows

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return counts, valid_rows, total_rows

        for row in reader:
            total_rows += 1
            # Skip partially-written lines while internvl.py is flushing/appending.
            if not row:
                continue

            row_has_any = False
            for category in categories:
                value = as_one_hot_int(row.get(category))
                counts[category] += value
                if value == 1:
                    row_has_any = True

            if row_has_any:
                valid_rows += 1

    return counts, valid_rows, total_rows


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    refresh = max(0.2, args.refresh)
    categories = args.categories

    running = True

    def stop_handler(signum, frame):  # type: ignore[no-untyped-def]
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    print(f"Monitoring category counts in {csv_path} (Ctrl+C to stop)")

    while running:
        counts, valid_rows, total_rows = read_counts(csv_path, categories)

        lines = ["\n=== InternVL Category Counts ==="]
        for name in categories:
            lines.append(f"{name:>8}: {counts[name]:,}")

        lines.append(f"counted rows (one-hot match): {valid_rows:,}")
        if args.show_total_rows:
            lines.append(f"total rows in csv: {total_rows:,}")
        lines.append(f"last update: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        sys.stdout.write("\033[2J\033[H")
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()

        time.sleep(refresh)

    print("Stopped monitor.")


if __name__ == "__main__":
    main()
