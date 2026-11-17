from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
from satpy import Scene
from tqdm import tqdm

from eumetsat_alignment_utils import (
    CENTER_LAT,
    CENTER_LON,
    HRV_CHANNEL,
    IR_CHANNEL,
    compact_time,
    load_sky_rgb_chw,
    parse_sat_filename_time,
    parse_sky_time,
    save_json,
)


def index_sky_images(data_root: Path, day: str) -> dict[dt.datetime, Path]:
    day_dir = data_root / day
    if not day_dir.exists():
        return {}
    return {parse_sky_time(path): path for path in sorted(day_dir.glob("*.jpg"))}


def index_satellite_files(data_root: Path, day: str) -> list[tuple[dt.datetime, Path]]:
    pattern = f"MSG4-SEVI-MSG15-0100-NA-{day}*.nat"
    return [(parse_sat_filename_time(path), path) for path in sorted(data_root.glob(pattern))]


def satellite_for_slot(
    indexed_satellites: list[tuple[dt.datetime, Path]],
    target_time: dt.datetime,
    slot_minutes: int,
) -> Path | None:
    slot_end = target_time + dt.timedelta(minutes=slot_minutes)
    for file_time, path in indexed_satellites:
        if target_time <= file_time < slot_end:
            return path
        if file_time >= slot_end:
            break
    return None


def crop_satellite_file(
    path: Path,
    center_lat: float,
    center_lon: float,
    ir_size: int,
    hrv_size: int,
) -> dict:
    scene = Scene(filenames=[str(path)], reader="seviri_l1b_native")
    scene.load([IR_CHANNEL, HRV_CHANNEL])

    crops = {}
    metadata = {"source_file": path.name}
    for name, size in [(IR_CHANNEL, ir_size), (HRV_CHANNEL, hrv_size)]:
        data = scene[name]
        area = data.attrs["area"]
        x_idx, y_idx = area.get_array_indices_from_lonlat(center_lon, center_lat)
        half = size // 2
        y0 = y_idx - half
        x0 = x_idx - half
        y1 = y0 + size
        x1 = x0 + size
        if y0 < 0 or x0 < 0 or y1 > data.shape[0] or x1 > data.shape[1]:
            raise ValueError(f"{name} crop is outside image bounds for {path.name}")

        crop = data.isel(y=slice(y0, y1), x=slice(x0, x1)).compute().values
        crops[name] = np.asarray(crop, dtype=np.float32)[None, :, :]
        metadata[f"{name}_start_time"] = data.attrs["start_time"].strftime("%Y%m%d%H%M%S")
        metadata[f"{name}_end_time"] = data.attrs["end_time"].strftime("%Y%m%d%H%M%S")
        metadata[f"{name}_wavelength"] = str(data.attrs["wavelength"])
        metadata[f"{name}_resolution_m"] = float(data.attrs["resolution"])
        metadata[f"{name}_center_pixel_x"] = int(x_idx)
        metadata[f"{name}_center_pixel_y"] = int(y_idx)

    return {
        "satellite_ir": crops[IR_CHANNEL],
        "satellite_hrv": crops[HRV_CHANNEL],
        "satellite_time": metadata[f"{IR_CHANNEL}_start_time"],
        "metadata": metadata,
    }


def valid_window_starts(
    sky_index: dict[dt.datetime, Path],
    satellite_index: list[tuple[dt.datetime, Path]],
    window_minutes: int,
    sky_step_minutes: int,
    satellite_step_minutes: int,
    window_stride_minutes: int,
    satellite_slot_minutes: int,
) -> list[dt.datetime]:
    if not sky_index or not satellite_index:
        return []

    sky_sequence_len = window_minutes // sky_step_minutes
    satellite_sequence_len = window_minutes // satellite_step_minutes

    starts = []
    current = min(sky_index)
    current = current.replace(
        minute=(current.minute // window_stride_minutes) * window_stride_minutes,
        second=0,
        microsecond=0,
    )
    while current < min(sky_index):
        current += dt.timedelta(minutes=window_stride_minutes)
    latest = max(sky_index) - dt.timedelta(minutes=window_minutes - sky_step_minutes)

    while current <= latest:
        sky_times = [current + dt.timedelta(minutes=sky_step_minutes * i) for i in range(sky_sequence_len)]
        satellite_times = [
            current + dt.timedelta(minutes=satellite_step_minutes * i)
            for i in range(satellite_sequence_len)
        ]
        sky_ok = all(t in sky_index for t in sky_times)
        sat_ok = all(
            satellite_for_slot(satellite_index, t, satellite_slot_minutes) is not None
            for t in satellite_times
        )
        if sky_ok and sat_ok:
            starts.append(current)
        current += dt.timedelta(minutes=window_stride_minutes)
    return starts


def build_day(
    data_root: Path,
    day: str,
    output_dir: Path,
    center_lat: float,
    center_lon: float,
    window_minutes: int,
    sky_step_minutes: int,
    satellite_step_minutes: int,
    window_stride_minutes: int,
    satellite_slot_minutes: int,
    ir_size: int,
    hrv_size: int,
) -> dict:
    sky_index = index_sky_images(data_root, day)
    satellite_index = index_satellite_files(data_root, day)
    sky_sequence_len = window_minutes // sky_step_minutes
    satellite_sequence_len = window_minutes // satellite_step_minutes
    starts = valid_window_starts(
        sky_index,
        satellite_index,
        window_minutes,
        sky_step_minutes,
        satellite_step_minutes,
        window_stride_minutes,
        satellite_slot_minutes,
    )

    day_summary = {
        "day": day,
        "sky_files": len(sky_index),
        "satellite_files": len(satellite_index),
        "sky_sequence_len": sky_sequence_len,
        "satellite_sequence_len": satellite_sequence_len,
        "valid_windows": len(starts),
        "saved_windows": 0,
        "output_files": [],
    }
    if not starts:
        return day_summary

    crop_cache: dict[Path, dict] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for start_time in tqdm(starts, desc=f"Build {day}", unit="window"):
        sky_times = [start_time + dt.timedelta(minutes=sky_step_minutes * i) for i in range(sky_sequence_len)]
        satellite_target_times = [
            start_time + dt.timedelta(minutes=satellite_step_minutes * i)
            for i in range(satellite_sequence_len)
        ]
        sat_paths = [
            satellite_for_slot(satellite_index, t, satellite_slot_minutes)
            for t in satellite_target_times
        ]
        assert all(path is not None for path in sat_paths)

        sky_seq = [load_sky_rgb_chw(sky_index[t]) for t in sky_times]
        hrv_seq = []
        ir_seq = []
        sat_times = []
        sat_files = []
        for sat_path in sat_paths:
            if sat_path not in crop_cache:
                crop_cache[sat_path] = crop_satellite_file(sat_path, center_lat, center_lon, ir_size, hrv_size)
            entry = crop_cache[sat_path]
            hrv_seq.append(entry["satellite_hrv"])
            ir_seq.append(entry["satellite_ir"])
            sat_times.append(entry["satellite_time"])
            sat_files.append(entry["metadata"]["source_file"])

        end_time = start_time + dt.timedelta(minutes=window_minutes)
        out_name = f"eumetsat_sirta_aligned_{compact_time(start_time)}_to_{compact_time(end_time)}.npz"
        out_path = output_dir / out_name
        np.savez_compressed(
            out_path,
            sky_image=np.stack(sky_seq, axis=0),
            satellite_hrv=np.stack(hrv_seq, axis=0),
            satellite_ir=np.stack(ir_seq, axis=0),
            sky_image_time=np.asarray([compact_time(t) for t in sky_times]),
            satellite_time=np.asarray(sat_times),
            satellite_target_time=np.asarray([compact_time(t) for t in satellite_target_times]),
            center_lat=np.asarray(center_lat, dtype=np.float32),
            center_lon=np.asarray(center_lon, dtype=np.float32),
            window_minutes=np.asarray(window_minutes, dtype=np.int32),
            sky_step_minutes=np.asarray(sky_step_minutes, dtype=np.int32),
            satellite_step_minutes=np.asarray(satellite_step_minutes, dtype=np.int32),
            window_stride_minutes=np.asarray(window_stride_minutes, dtype=np.int32),
            ir_channel=np.asarray(IR_CHANNEL),
            visible_channel=np.asarray(HRV_CHANNEL),
            source_satellite_files=np.asarray(sat_files),
            source_sky_files=np.asarray([sky_index[t].name for t in sky_times]),
        )
        day_summary["saved_windows"] += 1
        day_summary["output_files"].append(str(out_path))

    return day_summary


def validate_time_config(window_minutes: int, sky_step_minutes: int, satellite_step_minutes: int) -> None:
    if window_minutes <= 0 or sky_step_minutes <= 0 or satellite_step_minutes <= 0:
        raise ValueError("window and step minutes must be positive")
    if window_minutes % sky_step_minutes != 0:
        raise ValueError("window_minutes must be divisible by sky_step_minutes")
    if window_minutes % satellite_step_minutes != 0:
        raise ValueError("window_minutes must be divisible by satellite_step_minutes")


def default_output_dir_for_config(
    base_dir: Path,
    window_minutes: int,
    sky_step_minutes: int,
    satellite_step_minutes: int,
) -> Path:
    if window_minutes == 120 and sky_step_minutes == 10 and satellite_step_minutes == 10:
        return base_dir / "outputs" / "aligned_sequences"
    suffix = f"sky{sky_step_minutes}min_sat{satellite_step_minutes}min_win{window_minutes}min"
    return base_dir / "outputs" / f"aligned_sequences_{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dates", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--center-lat", type=float, default=CENTER_LAT)
    parser.add_argument("--center-lon", type=float, default=CENTER_LON)
    parser.add_argument("--window-minutes", type=int, default=120)
    parser.add_argument("--sky-step-minutes", type=int, default=10)
    parser.add_argument("--satellite-step-minutes", type=int, default=10)
    parser.add_argument("--window-stride-minutes", type=int, default=10)
    parser.add_argument("--satellite-slot-minutes", type=int, default=5)
    parser.add_argument("--ir-size", type=int, default=128)
    parser.add_argument("--hrv-size", type=int, default=256)
    parser.add_argument("--sequence-len", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--step-minutes", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.step_minutes is not None:
        args.sky_step_minutes = args.step_minutes
        args.satellite_step_minutes = args.step_minutes
    if args.sequence_len is not None:
        args.window_minutes = args.sequence_len * args.sky_step_minutes

    validate_time_config(args.window_minutes, args.sky_step_minutes, args.satellite_step_minutes)
    if args.window_stride_minutes <= 0 or args.satellite_slot_minutes <= 0:
        raise ValueError("window_stride_minutes and satellite_slot_minutes must be positive")

    if not args.data_root.is_dir():
        parser.error(f"Data root is not a directory: {args.data_root}")
    for day in args.dates:
        try:
            if len(day) != 8 or not day.isdigit():
                raise ValueError(day)
            dt.datetime.strptime(day, "%Y%m%d")
        except ValueError:
            parser.error(f"Invalid date {day!r}; use YYYYMMDD")
        if not (args.data_root / day).is_dir():
            parser.error(f"Sky image directory does not exist: {args.data_root / day}")
    if args.ir_size <= 0 or args.hrv_size <= 0:
        parser.error("Crop sizes must be positive")

    output_dir = args.output_dir or default_output_dir_for_config(
        Path(__file__).parent,
        args.window_minutes,
        args.sky_step_minutes,
        args.satellite_step_minutes,
    )

    summaries = []
    for day in args.dates:
        summaries.append(
            build_day(
                data_root=args.data_root,
                day=day,
                output_dir=output_dir,
                center_lat=args.center_lat,
                center_lon=args.center_lon,
                window_minutes=args.window_minutes,
                sky_step_minutes=args.sky_step_minutes,
                satellite_step_minutes=args.satellite_step_minutes,
                window_stride_minutes=args.window_stride_minutes,
                satellite_slot_minutes=args.satellite_slot_minutes,
                ir_size=args.ir_size,
                hrv_size=args.hrv_size,
            )
        )

    summary = {
        "data_root": str(args.data_root),
        "output_dir": str(output_dir),
        "center_lat": args.center_lat,
        "center_lon": args.center_lon,
        "window_minutes": args.window_minutes,
        "sky_step_minutes": args.sky_step_minutes,
        "satellite_step_minutes": args.satellite_step_minutes,
        "sky_sequence_len": args.window_minutes // args.sky_step_minutes,
        "satellite_sequence_len": args.window_minutes // args.satellite_step_minutes,
        "window_stride_minutes": args.window_stride_minutes,
        "satellite_slot_minutes": args.satellite_slot_minutes,
        "ir_channel": IR_CHANNEL,
        "visible_channel": HRV_CHANNEL,
        "days": summaries,
        "total_saved_windows": sum(item["saved_windows"] for item in summaries),
    }
    summary_name = "alignment_summary.json" if output_dir.name == "aligned_sequences" else f"{output_dir.name}_summary.json"
    summary_path = output_dir.parent / summary_name
    save_json(summary_path, summary)
    print(f"Saved summary: {summary_path}")
    print(f"Total saved windows: {summary['total_saved_windows']}")


if __name__ == "__main__":
    main()
