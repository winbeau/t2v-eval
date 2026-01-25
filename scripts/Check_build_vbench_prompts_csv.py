#!/usr/bin/env python3
"""
Build prompts_from_filenames.csv for VBench sampled videos.

Optionally validate that prompts match official VBench prompt lists.
"""

import argparse
import csv
import re
from pathlib import Path


def _load_official_prompts(suite: str) -> set[str]:
    if suite == "per_dimension":
        prompt_file = Path("third_party/VBench/prompts/all_dimension.txt")
    else:
        prompt_file = Path("third_party/VBench/prompts/all_category.txt")

    if not prompt_file.exists():
        return set()

    prompts = set()
    for line in prompt_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            prompts.add(line)
    return prompts


def _prompt_from_filename(stem: str) -> str:
    return re.sub(r"-\d+$", "", stem)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build prompt CSV for VBench sampled videos.")
    parser.add_argument(
        "--root",
        default="./hf/VBench_sampled_video",
        help="Root directory of downloaded videos",
    )
    parser.add_argument(
        "--suite",
        default="per_dimension",
        choices=["per_dimension", "per_category"],
        help="Suite folder to scan",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: {root}/prompts_from_filenames.csv)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate prompts against official VBench prompt list",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if validation finds unknown prompts",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    suite_dir = root / args.suite
    if not suite_dir.exists():
        raise SystemExit(f"Suite directory not found: {suite_dir}")

    output = Path(args.output) if args.output else (root / "prompts_from_filenames.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in suite_dir.rglob("*.mp4"):
        rel = p.relative_to(suite_dir)
        group = rel.parts[0] if rel.parts else "unknown"
        video_id = p.stem
        prompt = _prompt_from_filename(video_id)
        rows.append({
            "video_id": video_id,
            "prompt": prompt,
            "group": group,
            "video_path": str(p),
        })

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "prompt", "group", "video_path"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {output} (rows: {len(rows)})")

    if args.validate:
        official = _load_official_prompts(args.suite)
        if not official:
            print("Official prompt list not found; skip validation.")
            return

        prompts_in_data = {r["prompt"] for r in rows}
        unknown = sorted(prompts_in_data - official)
        missing = sorted(official - prompts_in_data)

        print(f"Validation: prompts_in_data={len(prompts_in_data)}, official={len(official)}")
        print(f"Unknown prompts: {len(unknown)}")
        print(f"Missing official prompts: {len(missing)}")

        if unknown:
            print("Example unknown prompts:")
            for p in unknown[:10]:
                print("  -", p)
        if missing:
            print("Example missing official prompts:")
            for p in missing[:10]:
                print("  -", p)

        if args.strict and unknown:
            raise SystemExit("Unknown prompts detected.")


if __name__ == "__main__":
    main()
