"""Filename, image and JSON helpers used by the sequence builder."""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


CENTER_LAT = 48.713
CENTER_LON = 2.208
IR_CHANNEL = "IR_108"
HRV_CHANNEL = "HRV"
TIME_FMT = "%Y%m%d%H%M%S"
SAT_NAME_RE = re.compile(r"-(\d{14})\.\d+Z-")


def parse_sky_time(path: Path) -> dt.datetime:
    return dt.datetime.strptime(path.name[:14], TIME_FMT)


def parse_sat_filename_time(path: Path) -> dt.datetime:
    match = SAT_NAME_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse satellite timestamp from {path.name}")
    return dt.datetime.strptime(match.group(1), TIME_FMT)


def compact_time(t: dt.datetime) -> str:
    return t.strftime(TIME_FMT)


def load_sky_rgb_chw(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
