#!/usr/bin/env python3
"""Render a heightmap as height/hillshade/slope triptych."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demforge.metrics import terrain_metrics
from demforge.render import load_height_array, render_triptych


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--metrics", default=None)
    args = parser.parse_args()

    height = load_height_array(args.input)
    render_triptych(height, args.out, title=Path(args.input).name)

    metrics = terrain_metrics(height)
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

    if args.metrics:
        import json

        Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
