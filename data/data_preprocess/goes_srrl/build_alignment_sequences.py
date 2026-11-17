"""Build aligned sequence NPZ files from already-preprocessed SRRL and GOES data."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from alignment import build_spatiotemporal_sequence_alignment_records
from data_io import DEFAULT_ALIGNMENT_YEAR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--srrl-aberration-root", type=Path, required=True,
                        help="Root of corrected SRRL JPGs in YYYYMMDD folders.")
    parser.add_argument("--srrl-cut-root", type=Path, required=True,
                        help="Root of cropped SRRL JPGs in YYYYMMDD folders.")
    parser.add_argument("--goes-year-root", type=Path, required=True,
                        help="One year's GOES files, in day-of-year folders (1 or 001).")
    parser.add_argument("--output-dir", "--sequence-output-dir", dest="output_dir", type=Path,
                        default=Path("outputs/aligned_sequences"),
                        help="Output NPZ directory, relative to the current working directory.")
    parser.add_argument("--year", type=int, default=DEFAULT_ALIGNMENT_YEAR,
                        help="SRRL year to select; must match --goes-year-root (default: 2021).")
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--srrl-interval-minutes", type=int, default=10)
    parser.add_argument("--goes-interval-minutes", type=int, default=5,
                        help="Target sampling interval; GOES coverage remains a fixed 5 minutes.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of day workers; use 1 for a first run (default: 4).")
    args = parser.parse_args()
    for name in ("srrl_aberration_root", "srrl_cut_root", "goes_year_root"):
        if not getattr(args, name).is_dir():
            parser.error(f"--{name.replace('_', '-')} is not a directory: {getattr(args, name)}")
    for name in ("sequence_length", "srrl_interval_minutes", "goes_interval_minutes", "workers"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 1000 <= args.year <= 9999:
        parser.error("--year must be a four-digit year")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=str(args.output_dir / "build_alignment_sequences.log"),
                        level=logging.INFO, format="%(message)s")
    build_spatiotemporal_sequence_alignment_records(
        srrl_aberration_root=args.srrl_aberration_root,
        srrl_cut_root=args.srrl_cut_root,
        goes_root=args.goes_year_root,
        output_dir=args.output_dir,
        year=args.year,
        srrl_sequence_length=args.sequence_length,
        srrl_interval_minutes=args.srrl_interval_minutes,
        goes_interval_minutes=args.goes_interval_minutes,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
