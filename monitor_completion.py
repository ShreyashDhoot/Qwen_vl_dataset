#!/usr/bin/env python
"""Live terminal progress monitor for CSV generation.

Tracks processed rows toward a target and shows a continuously updating bar
with ETA based on observed throughput.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path

DEFAULT_CSV = "internvl_predictions.csv"
DEFAULT_TARGET = 79000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor CSV progress with live ETA.")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="CSV file to monitor.")
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET, help="Target number of samples.")
    parser.add_argument("--refresh", type=float, default=1.0, help="Refresh interval in seconds.")
    parser.add_argument("--bar-width", type=int, default=40, help="Progress bar width.")
    parser.add_argument(
        "--window",
        type=int,
        default=120,
        help="Seconds of history used to estimate throughput and ETA.",
    )
    return parser.parse_args()


def count_rows(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    # Count data rows (minus header line if present).
    with csv_path.open("r", encoding="utf-8", errors="ignore") as handle:
        line_count = sum(1 for _ in handle)
    return max(0, line_count - 1)


def human_time(seconds: float | None) -> str:
    if seconds is None or seconds == float("inf"):
        return "--:--:--"
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def render_bar(done: int, total: int, width: int) -> str:
    if total <= 0:
        total = 1
    frac = min(max(done / total, 0.0), 1.0)
    filled = int(frac * width)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {frac * 100:6.2f}%"


def terminal_width(default: int = 120) -> int:
    try:
        return os.get_terminal_size().columns
    except OSError:
        return default


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    target = max(1, args.target)
    refresh = max(0.2, args.refresh)
    window_seconds = max(10, args.window)

    running = True

    def stop_handler(signum, frame):  # type: ignore[no-untyped-def]
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    history: deque[tuple[float, int]] = deque()
    started_at = time.time()

    print(f"Monitoring {csv_path} toward target={target} (Ctrl+C to stop)")

    while running:
        now = time.time()
        done = count_rows(csv_path)
        done_clamped = min(done, target)

        history.append((now, done))
        while history and (now - history[0][0]) > window_seconds:
            history.popleft()

        speed = 0.0
        if len(history) >= 2:
            t0, c0 = history[0]
            t1, c1 = history[-1]
            dt = max(t1 - t0, 1e-6)
            speed = max(0.0, (c1 - c0) / dt)

        remaining = max(0, target - done)
        eta = (remaining / speed) if speed > 1e-9 else None

        elapsed = now - started_at
        bar = render_bar(done_clamped, target, args.bar_width)
        status = (
            f"{bar}  done={done:,}/{target:,}  left={remaining:,}"
            f"  speed={speed:,.2f}/s  elapsed={human_time(elapsed)}  eta={human_time(eta)}"
        )

        # Fit to terminal width and refresh in place.
        width = terminal_width()
        if len(status) > width - 1:
            status = status[: width - 1]
        sys.stdout.write("\r" + status.ljust(max(1, width - 1)))
        sys.stdout.flush()

        if done >= target:
            break

        time.sleep(refresh)

    print()
    final_done = count_rows(csv_path)
    if final_done >= target:
        print(f"Target reached: {final_done:,}/{target:,}")
    else:
        print(f"Stopped at: {final_done:,}/{target:,}")


if __name__ == "__main__":
    main()
