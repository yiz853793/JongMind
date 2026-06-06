"""Command line entry point for replay dataset construction."""

from __future__ import annotations

import argparse

from jongmind.replay_dataset import build_dataset_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Build imitation samples from hand history JSONL.")
    parser.add_argument("--input", required=True, help="Input hand_history JSONL path, for example logs/hands.jsonl.")
    parser.add_argument("--output", required=True, help="Output PyTorch dataset path, for example data/imitation.pt.")
    args = parser.parse_args()

    payload = build_dataset_file(args.input, args.output)
    counts = ", ".join(f"{name}={count}" for name, count in payload["sample_type_counts"].items())
    print(f"saved {payload['sample_count']} samples to {args.output} ({counts})")


if __name__ == "__main__":
    main()
