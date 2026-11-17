"""Original cached, per-day SRRL/GOES sequence construction."""
from __future__ import annotations

import datetime as dt
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from tqdm import tqdm

from data_io import (
    DEFAULT_ALIGNMENT_YEAR, GOESRadianceProduct, MST,
    SRRLSkyImage, SRRL_ALIGNED_HALF_PIXEL, SRRL_LAT, SRRL_LON,
    estimate_region_extent_km, find_goes_product_covering,
    load_goes_products_for_day, resolve_goes_day_dir,
)


def print_log(msg: str) -> None:
    logging.info(msg)
    try:
        print(msg)
    except Exception:
        pass


@dataclass(frozen=True)
class SRRLSequenceFrame:
    aberration_11: Path
    aberration_12: Path
    cut_11: Path
    cut_12: Path
    local_dt: dt.datetime
    utc_dt: dt.datetime


def _load_day_intervals(
    day_of_year: int,
    goes_root: Path,
    interval_cache: Dict[int, List[GOESRadianceProduct]],
    day_label: Optional[str] = None,
) -> List[GOESRadianceProduct]:
    intervals = interval_cache.get(day_of_year)
    if intervals is None:
        goes_day_dir = resolve_goes_day_dir(goes_root, day_of_year)
        if not goes_day_dir:
            suffix = f" (SRRL date {day_label})" if day_label else ""
            print_log(
                f"[WARN] Missing GOES directory for DOY {day_of_year}"
                f"{suffix}: root {goes_root}"
            )
            intervals = []
        else:
            intervals = load_goes_products_for_day(goes_day_dir)
        interval_cache[day_of_year] = intervals
    return intervals


def generate_time_list(
    start_time: Union[str, dt.datetime],
    time_interval: int,
    frames_len: int,
) -> List[Union[str, dt.datetime]]:
    """Generate timestamps every time_interval minutes."""
    is_str_input = isinstance(start_time, str)
    if is_str_input:
        base = dt.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    else:
        base = start_time
    times = [base + dt.timedelta(minutes=i * time_interval) for i in range(frames_len)]
    if is_str_input:
        return [t.strftime("%Y-%m-%d %H:%M:%S") for t in times]
    return times


def calculate_possible_sequence_starts(
    sorted_times: Sequence[dt.datetime],
    time_zones: Sequence[Tuple[dt.datetime, dt.datetime]],
    frames_len: int,
    time_interval: int,
) -> List[int]:
    """Return indices whose windows form consecutive timestamps within time_zones."""
    if frames_len <= 0 or len(sorted_times) < frames_len:
        return []
    possible: List[int] = []
    for idx in range(len(sorted_times) - frames_len + 1):
        start_time = sorted_times[idx]
        end_time = sorted_times[idx + frames_len - 1]
        if not any(zone_start <= start_time and end_time <= zone_end for zone_start, zone_end in time_zones):
            continue
        expected = generate_time_list(start_time, time_interval, frames_len)
        if all(sorted_times[idx + offset] == expected[offset] for offset in range(frames_len)):
            possible.append(idx)
    return possible


def save_sequence_alignment_record(
    output_dir: Path,
    srrl_start_local_time: str,
    srrl_end_local_time: str,
    payload: Dict[str, np.ndarray],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_name = (
        f"goes_srrl_sequence_aligned_{srrl_start_local_time}_to_{srrl_end_local_time}.npz"
    )
    output_path = output_dir / file_name
    np.savez_compressed(output_path, **payload)
    return output_path


def _process_one_day_sequence(
    day_dir: Path,
    srrl_cut_root: Path,
    goes_root: Path,
    output_dir: Path,
    srrl_sequence_length: int,
    srrl_interval_minutes: int,
    goes_interval_minutes: int,
    center_lat: float,
    center_lon: float,
    half_px: int,
) -> Tuple[int, int, int, int]:
    """
    返回 (saved_records, skipped_due_to_srrl, skipped_due_to_goes, missing_pairs_total)
    其中 missing_pairs_total = missing_dual_exposures + missing_cut_exposures
    """
    try:
        cut_day_dir = srrl_cut_root / day_dir.name
        if not cut_day_dir.exists():
            print_log(f"[WARN] Missing cut directory for {day_dir.name}: {cut_day_dir}")
            return 0, 0, 0, 0

        # 收集当日所有帧（仅成功的双曝光且存在 cut）
        frames: List[SRRLSequenceFrame] = []
        missing_dual_exposures = 0
        missing_cut_exposures = 0

        for aberration_11_path in sorted(day_dir.glob("*_11.jpg")):
            counterpart_name = aberration_11_path.name.replace("_11.jpg", "_12.jpg")
            aberration_12_path = day_dir / counterpart_name
            if not aberration_12_path.exists():
                print_log(f"[WARN] Missing aberration _12 exposure for {aberration_11_path.name}")
                missing_dual_exposures += 1
                continue

            cut_11_path = cut_day_dir / aberration_11_path.name
            cut_12_path = cut_day_dir / counterpart_name
            if not cut_11_path.exists() or not cut_12_path.exists():
                if not cut_11_path.exists():
                    print_log(f"[WARN] Missing cut _11 image: {cut_11_path}")
                if not cut_12_path.exists():
                    print_log(f"[WARN] Missing cut _12 image: {cut_12_path}")
                missing_cut_exposures += 1
                continue

            capture = SRRLSkyImage(aberration_11_path)
            frames.append(
                SRRLSequenceFrame(
                    aberration_11=aberration_11_path,
                    aberration_12=aberration_12_path,
                    cut_11=cut_11_path,
                    cut_12=cut_12_path,
                    local_dt=capture.capture_time_mst(),
                    utc_dt=capture.capture_time_utc(),
                )
            )

        frames.sort(key=lambda f: f.local_dt)
        if len(frames) < srrl_sequence_length:
            return 0, 0, 0, (missing_dual_exposures + missing_cut_exposures)

        # 本地日时间窗口
        day_start_local = dt.datetime.strptime(day_dir.name, "%Y%m%d").replace(tzinfo=MST)
        day_end_local = day_start_local + dt.timedelta(days=1)
        time_zones = [(day_start_local, day_end_local)]

        local_times = [f.local_dt for f in frames]
        possible_starts = calculate_possible_sequence_starts(
            local_times, time_zones, srrl_sequence_length, srrl_interval_minutes
        )
        if not possible_starts:
            return 0, 0, 0, (missing_dual_exposures + missing_cut_exposures)

        # ====== 预计算：SRRL 与 GOES 使用计数（用于引用计数缓存）======
        srrl_use_count: Dict[Path, int] = {}
        goes_use_count: Dict[Path, int] = {}
        goes_interval = dt.timedelta(minutes=goes_interval_minutes)
        goes_interval_cache: Dict[int, List[GOESRadianceProduct]] = {}

        def _incr(d: Dict[Path, int], k: Path, n: int = 1):
            d[k] = d.get(k, 0) + n

        # 遍历所有序列，累计 SRRL 与 GOES 需求
        for start_idx in possible_starts:
            seq = frames[start_idx : start_idx + srrl_sequence_length]
            # SRRL 4类图像
            for fr in seq:
                _incr(srrl_use_count, fr.aberration_11)
                _incr(srrl_use_count, fr.aberration_12)
                _incr(srrl_use_count, fr.cut_11)
                _incr(srrl_use_count, fr.cut_12)

            # 需要的 GOES 时间点（UTC）
            current_time = seq[0].utc_dt
            final_time = seq[-1].utc_dt + goes_interval
            while current_time <= final_time:
                doy = current_time.timetuple().tm_yday
                intervals = _load_day_intervals(doy, goes_root, goes_interval_cache, current_time.strftime("%Y-%m-%d"))
                if intervals:
                    match = find_goes_product_covering(intervals, current_time)
                    if match and match.path:
                        _incr(goes_use_count, match.path)
                current_time += goes_interval

        # ====== 缓存容器 ======
        srrl_cache: Dict[Path, np.ndarray] = {}
        goes_cache: Dict[Path, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        def _load_srrl_cached(p: Path) -> Optional[np.ndarray]:
            arr = srrl_cache.get(p)
            if arr is None:
                try:
                    arr = SRRLSkyImage(p).load_rgb().astype(np.uint8, copy=False)
                    srrl_cache[p] = arr
                except Exception as exc:
                    print_log(f"[WARN] Failed to read SRRL {p.name}: {exc}")
                    return None
            # 使用后减计数，如为0则释放
            srrl_use_count[p] -= 1
            if srrl_use_count[p] <= 0:
                srrl_cache.pop(p, None)
            return arr

        def _load_goes_crop_cached(ts: dt.datetime) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, str, str, str]]:
            doy = ts.timetuple().tm_yday
            intervals = _load_day_intervals(doy, goes_root, goes_interval_cache, ts.strftime("%Y-%m-%d"))
            if not intervals:
                return None
            match = find_goes_product_covering(intervals, ts)
            if match is None:
                return None
            path = match.path
            entry = goes_cache.get(path)
            if entry is None:
                try:
                    with GOESRadianceProduct(path) as product:
                        region_rad, region_lat, region_lon = product.crop_region_by_pixels(
                            center_lat=center_lat,
                            center_lon=center_lon,
                            half_size_px=half_px,
                        )
                    entry = (region_rad.astype(np.float32, copy=False),
                             region_lat.astype(np.float32, copy=False),
                             region_lon.astype(np.float32, copy=False))
                    goes_cache[path] = entry
                except Exception as exc:
                    print_log(f"[WARN] Failed to crop GOES for {path.name}: {exc}")
                    return None
            # 组成时间字符串
            start_str = match.start_time.strftime("%Y%m%d%H%M%S") if match.start_time else ""
            end_str = match.end_time.strftime("%Y%m%d%H%M%S") if match.end_time else ""
            fake_end_str = match.fake_end_time.strftime("%Y%m%d%H%M%S") if match.fake_end_time else ""
            # 使用后减计数，如为0则释放
            goes_use_count[path] -= 1
            if goes_use_count[path] <= 0:
                goes_cache.pop(path, None)
            return entry[0], entry[1], entry[2], start_str, end_str, fake_end_str

        # ====== 正式构建并保存序列 ======
        saved_records = 0
        skipped_due_to_srrl = 0
        skipped_due_to_goes = 0

        for start_idx in possible_starts:
            seq = frames[start_idx : start_idx + srrl_sequence_length]

            srrl_img_aberration_11_seq: List[np.ndarray] = []
            srrl_img_aberration_12_seq: List[np.ndarray] = []
            srrl_img_cut_11_seq: List[np.ndarray] = []
            srrl_img_cut_12_seq: List[np.ndarray] = []
            srrl_local_times: List[str] = []
            srrl_utc_times: List[str] = []

            sequence_has_missing = False
            for fr in seq:
                ab11 = _load_srrl_cached(fr.aberration_11)
                ab12 = _load_srrl_cached(fr.aberration_12)
                ct11 = _load_srrl_cached(fr.cut_11)
                ct12 = _load_srrl_cached(fr.cut_12)
                if ab11 is None or ab12 is None or ct11 is None or ct12 is None:
                    sequence_has_missing = True
                    break
                srrl_img_aberration_11_seq.append(ab11)
                srrl_img_aberration_12_seq.append(ab12)
                srrl_img_cut_11_seq.append(ct11)
                srrl_img_cut_12_seq.append(ct12)
                srrl_local_times.append(fr.local_dt.strftime("%Y%m%d%H%M%S"))
                srrl_utc_times.append(fr.utc_dt.strftime("%Y%m%d%H%M%S"))

            if sequence_has_missing:
                skipped_due_to_srrl += 1
                continue

            # GOES 时间序列
            goes_imgs_seq: List[np.ndarray] = []
            goes_requested_times: List[str] = []
            goes_start_times: List[str] = []
            goes_end_times: List[str] = []
            goes_fake_end_times: List[str] = []

            goes_region_lat: Optional[np.ndarray] = None
            goes_region_lon: Optional[np.ndarray] = None
            goes_bounds_lat: Optional[Tuple[float, float]] = None
            goes_bounds_lon: Optional[Tuple[float, float]] = None
            goes_height_km: Optional[float] = None
            goes_width_km: Optional[float] = None

            current_time = seq[0].utc_dt
            final_time = seq[-1].utc_dt + goes_interval
            sequence_goes_missing = False
            while current_time <= final_time:
                out = _load_goes_crop_cached(current_time)
                if out is None:
                    sequence_goes_missing = True
                    break
                region_rad, region_lat, region_lon, s_str, e_str, fe_str = out
                goes_imgs_seq.append(region_rad)
                goes_requested_times.append(current_time.strftime("%Y%m%d%H%M%S"))
                goes_start_times.append(s_str)
                goes_end_times.append(e_str)
                goes_fake_end_times.append(fe_str)

                if goes_region_lat is None or goes_region_lon is None:
                    goes_region_lat = region_lat
                    goes_region_lon = region_lon
                    goes_bounds_lat = (float(np.nanmin(region_lat)), float(np.nanmax(region_lat)))
                    goes_bounds_lon = (float(np.nanmin(region_lon)), float(np.nanmax(region_lon)))
                    goes_height_km, goes_width_km = estimate_region_extent_km(region_lat, region_lon)

                current_time += goes_interval

            if sequence_goes_missing or goes_region_lat is None or goes_region_lon is None:
                skipped_due_to_goes += 1
                continue

            payload = {
                "srrl_img_aberration_11_seq": np.stack(srrl_img_aberration_11_seq, axis=0),
                "srrl_img_aberration_12_seq": np.stack(srrl_img_aberration_12_seq, axis=0),
                "srrl_img_cut_11_seq": np.stack(srrl_img_cut_11_seq, axis=0),
                "srrl_img_cut_12_seq": np.stack(srrl_img_cut_12_seq, axis=0),
                "srrl_local_times_seq": np.array(srrl_local_times),
                "srrl_utc_times_seq": np.array(srrl_utc_times),
                "goes_imgs_seq": np.stack(goes_imgs_seq, axis=0),
                "goes_requested_utc_times": np.array(goes_requested_times),
                "goes_utc_start_times": np.array(goes_start_times),
                "goes_utc_end_times": np.array(goes_end_times),
                "goes_utc_fake_end_times": np.array(goes_fake_end_times),
                "goes_lats": goes_region_lat,
                "goes_lons": goes_region_lon,
                "goes_lats_min": np.array(np.float32(goes_bounds_lat[0])),
                "goes_lats_max": np.array(np.float32(goes_bounds_lat[1])),
                "goes_lons_min": np.array(np.float32(goes_bounds_lon[0])),
                "goes_lons_max": np.array(np.float32(goes_bounds_lon[1])),
                "goes_resolution_h_km": np.array(np.float32(goes_height_km)),
                "goes_resolution_w_km": np.array(np.float32(goes_width_km)),
            }

            start_local = seq[0].local_dt.strftime("%Y%m%d%H%M%S")
            end_local = seq[-1].local_dt.strftime("%Y%m%d%H%M%S")
            output_path = save_sequence_alignment_record(
                output_dir=output_dir,
                srrl_start_local_time=start_local,
                srrl_end_local_time=end_local,
                payload=payload,
            )
            saved_records += 1
            print_log(
                f"[INFO] Saved sequence record: {day_dir.name} "
                f"{seq[0].aberration_11.name} -> {output_path.name}"
            )

        return saved_records, skipped_due_to_srrl, skipped_due_to_goes, (missing_dual_exposures + missing_cut_exposures)

    except Exception as exc:
        print_log(f"[ERROR] Failed day {day_dir.name}: {exc}")
        return 0, 0, 0, 0


def build_spatiotemporal_sequence_alignment_records(
    srrl_aberration_root: Path,
    srrl_cut_root: Path,
    goes_root: Path,
    output_dir: Path,
    year: int = DEFAULT_ALIGNMENT_YEAR,
    srrl_sequence_length: int = 12,
    srrl_interval_minutes: int = 10,
    goes_interval_minutes: int = 5,
    workers: int = 1,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    pattern = f"{year}*"
    day_dirs = [p for p in sorted(srrl_aberration_root.glob(pattern)) if p.is_dir()]
    if not day_dirs:
        print_log(f"[WARN] No SRRL aberration folders matching {pattern}")
        return

    total_saved = 0
    total_skip_srrl = 0
    total_skip_goes = 0
    total_missing_pairs = 0

    if workers and workers > 1:
        # 并行按天处理
        with ProcessPoolExecutor(max_workers=workers) as ex, tqdm(total=len(day_dirs), desc=f"Sequence Days {year}", unit="day") as pbar:
            futs = [
                ex.submit(
                    _process_one_day_sequence,
                    day_dir,
                    srrl_cut_root,
                    goes_root,
                    output_dir,
                    srrl_sequence_length,
                    srrl_interval_minutes,
                    goes_interval_minutes,
                    SRRL_LAT,
                    SRRL_LON,
                    SRRL_ALIGNED_HALF_PIXEL,
                )
                for day_dir in day_dirs
            ]
            for fut in as_completed(futs):
                saved, skip_srrl, skip_goes, miss_pairs = fut.result()
                total_saved += saved
                total_skip_srrl += skip_srrl
                total_skip_goes += skip_goes
                total_missing_pairs += miss_pairs
                pbar.update(1)
    else:
        # 顺序处理（仍使用缓存）
        for day_dir in tqdm(day_dirs, desc=f"Sequence Days {year}", unit="day"):
            saved, skip_srrl, skip_goes, miss_pairs = _process_one_day_sequence(
                day_dir,
                srrl_cut_root,
                goes_root,
                output_dir,
                srrl_sequence_length,
                srrl_interval_minutes,
                goes_interval_minutes,
                SRRL_LAT,
                SRRL_LON,
                SRRL_ALIGNED_HALF_PIXEL,
            )
            total_saved += saved
            total_skip_srrl += skip_srrl
            total_skip_goes += skip_goes
            total_missing_pairs += miss_pairs

    print_log(
        "[INFO] Completed sequence dataset build with "
        f"{total_saved} records. Skipped due to SRRL reads: {total_skip_srrl}. "
        f"Skipped due to GOES coverage: {total_skip_goes}. "
        f"Missing pairs (aberration/cut): {total_missing_pairs}."
    )
