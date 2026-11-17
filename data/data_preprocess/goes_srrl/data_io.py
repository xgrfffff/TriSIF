"""Image readers, GOES geometry and original timestamp matching helpers."""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from netCDF4 import Dataset
from PIL import Image
from pyproj import Geod, Proj

GEOD = Geod(ellps="GRS80")


DEFAULT_ALIGNMENT_YEAR = 2021
MST = dt.timezone(dt.timedelta(hours=-7), name="MST")
SRRL_LAT = 39.7423
SRRL_LON = -105.1785
SRRL_ALIGNED_HALF_PIXEL = 64
GOES_INTERVAL_REGEX = re.compile(r"_s(\d{14})_e(\d{14})")
GOES_FAKE_INTERVAL_MINUTES = 5


@dataclass
class GOESRadianceProduct:
    """Convenience wrapper for GOES ABI L1b radiance products."""

    path: Path
    start_time: Optional[dt.datetime] = None
    end_time: Optional[dt.datetime] = None
    fake_end_time: Optional[dt.datetime] = None
    _dataset: Optional[Dataset] = None
    _latlon_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def __enter__(self) -> "GOESRadianceProduct":
        self.open()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def open(self) -> Dataset:
        if self._dataset is None:
            self._dataset = Dataset(self.path)
        return self._dataset

    def close(self) -> None:
        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None

    def get_radiance_image(self) -> np.ndarray:
        dataset = self.open()
        radiance_var = dataset.variables["Rad"]
        data = radiance_var[:]
        if np.ma.isMaskedArray(data):
            data = data.filled(np.nan)
        return np.asarray(data, dtype=np.float32)

    def _build_proj(self) -> Proj:
        dataset = self.open()
        proj_var = dataset.variables["goes_imager_projection"]
        return Proj(
            proj="geos",
            h=proj_var.perspective_point_height,
            lon_0=proj_var.longitude_of_projection_origin,
            sweep=getattr(proj_var, "sweep_angle_axis", "x"),
            a=proj_var.semi_major_axis,
            b=proj_var.semi_minor_axis,
        )

    def get_lat_lon_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._latlon_cache is None:
            dataset = self.open()
            proj_var = dataset.variables["goes_imager_projection"]
            proj = self._build_proj()
            x = dataset.variables["x"][:]
            y = dataset.variables["y"][:]
            xx, yy = np.meshgrid(x, y)
            scale = proj_var.perspective_point_height
            xx_m = xx * scale
            yy_m = yy * scale
            lon, lat = proj(xx_m, yy_m, inverse=True)
            self._latlon_cache = (
                np.asarray(lat, dtype=np.float32),
                np.asarray(lon, dtype=np.float32),
            )
        return self._latlon_cache

    def _single_point_lat_lon_2_pixel(self, target_lat: float, target_lon: float) -> Tuple[int, int]:
        """
        返回最接近目标经纬度的像素行列索引 (i, j)。
        使用经纬度上近似等效距离的度量（对经度差乘以 cos(lat) 进行缩放）。
        """
        lats, lons = self.get_lat_lon_grid()
        valid = np.isfinite(lats) & np.isfinite(lons)
        if not valid.any():
            raise ValueError("No valid lat/lon in GOES grid.")
        # 近似地表等距：对经度差乘以 cos(target_lat)
        cos_lat = np.cos(np.deg2rad(target_lat))
        d2 = np.where(
            valid,
            (lats - target_lat) ** 2 + ((lons - target_lon) * cos_lat) ** 2,
            np.inf,
        )
        k = int(np.argmin(d2))
        return np.unravel_index(k, lats.shape)

    def crop_region_by_pixels(
        self,
        center_lat: float,
        center_lon: float,
        half_size_px: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        以(center_lat, center_lon)附近的最近像素为中心，上下左右各 half_size_px 像素，
        返回严格大小为 (2*half_size_px)×(2*half_size_px) 的 radiance/lat/lon 子阵列。
        会在接近边界时平移窗口以保持固定尺寸。
        """
        rad = self.get_radiance_image()
        lats, lons = self.get_lat_lon_grid()
        H, W = rad.shape
        win = half_size_px * 2
        if H < win or W < win:
            raise ValueError(f"Full image too small ({H}x{W}) for window {win}x{win}.")

        i0, j0 = self._single_point_lat_lon_2_pixel(center_lat, center_lon)

        # 先按中心切窗，再在边界处平移，保证大小恒定
        r_start = max(0, i0 - half_size_px)
        c_start = max(0, j0 - half_size_px)
        # 若越界，向回挪动
        r_start = min(r_start, H - win)
        c_start = min(c_start, W - win)

        r_end = r_start + win
        c_end = c_start + win

        return (
            rad[r_start:r_end, c_start:c_end],
            lats[r_start:r_end, c_start:c_end],
            lons[r_start:r_end, c_start:c_end],
        )


@dataclass
class SRRLSkyImage:
    """Wrapper for SRRL ground-based RGB sky images."""

    path: Path

    def load_rgb(self) -> np.ndarray:
        with Image.open(self.path) as img:
            rgb = img.convert("RGB")
            return np.asarray(rgb, dtype=np.uint8)

    def capture_time_mst(self) -> dt.datetime:
        stem = self.path.stem
        timestamp_token = stem.split("_")[0]
        local_naive = dt.datetime.strptime(timestamp_token, "%Y%m%d%H%M%S")
        return local_naive.replace(tzinfo=MST)

    def capture_time_utc(self) -> dt.datetime:
        return self.capture_time_mst().astimezone(dt.timezone.utc)


def parse_goes_time_code(code: str) -> dt.datetime:
    """Convert a GOES s/e token (YYYYDDDHHMMSSd) to timezone-aware UTC."""
    if len(code) != 14:
        raise ValueError(f"Unexpected GOES time token length: {code!r}")
    year = int(code[0:4])
    day_of_year = int(code[4:7])
    hour = int(code[7:9])
    minute = int(code[9:11])
    second = int(code[11:13])
    tenth = int(code[13])
    base = dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(
        days=day_of_year - 1
    )
    return base + dt.timedelta(
        hours=hour, minutes=minute, seconds=second, microseconds=tenth * 100000
    )


def build_goes_product_stub(path: Path) -> GOESRadianceProduct:
    """Create a GOESRadianceProduct with coverage metadata extracted from filename."""
    match = GOES_INTERVAL_REGEX.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse GOES filename interval from {path.name}")
    start_code, end_code = match.groups()

    start_dt = parse_goes_time_code(start_code)
    end_dt = parse_goes_time_code(end_code)
    fake_end = start_dt + dt.timedelta(minutes=GOES_FAKE_INTERVAL_MINUTES)

    return GOESRadianceProduct(
        path=path,
        start_time=start_dt,
        end_time=end_dt,
        fake_end_time=fake_end,
    )


def load_goes_products_for_day(day_dir: Path) -> List[GOESRadianceProduct]:
    """Parse every .nc file under the provided GOES day directory."""
    intervals: List[GOESRadianceProduct] = []
    for nc_file in sorted(day_dir.glob("*.nc")):
        try:
            intervals.append(build_goes_product_stub(nc_file))
        except ValueError as exc:
            print(f"[WARN] {exc}")
    return intervals


def find_goes_product_covering(
    intervals: List[GOESRadianceProduct], timestamp: dt.datetime
) -> Optional[GOESRadianceProduct]:
    """Return the GOES granule covering the requested timestamp, if any."""
    for interval in intervals:
        if interval.start_time and interval.fake_end_time and interval.start_time <= timestamp < interval.fake_end_time:
            return interval
    return None


def resolve_goes_day_dir(goes_root: Path, day_of_year: int) -> Optional[Path]:
    """Locate the GOES subdirectory for a day-of-year (supports 1/3-digit naming)."""
    candidates = [goes_root / f"{day_of_year}", goes_root / f"{day_of_year:03d}"]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def estimate_region_extent_km(
    region_lats: np.ndarray, region_lons: np.ndarray
) -> Tuple[float, float]:
    """Estimate approximate height/width (km) of a GOES sub-region."""
    h, w = region_lats.shape
    valid = np.isfinite(region_lats) & np.isfinite(region_lons)

    width_km = float("nan")
    row_counts = np.sum(valid, axis=1)
    if row_counts.max() > 1:
        r = int(np.argmax(row_counts))
        cols = np.where(valid[r])[0]
        c0, c1 = int(cols.min()), int(cols.max())
        dist_m = abs(
            GEOD.inv(
                region_lons[r, c0],
                region_lats[r, c0],
                region_lons[r, c1],
                region_lats[r, c1],
            )[2]
        )
        width_km = dist_m / 1000.0

    height_km = float("nan")
    col_counts = np.sum(valid, axis=0)
    if col_counts.max() > 1:
        c = int(np.argmax(col_counts))
        rows = np.where(valid[:, c])[0]
        r0, r1 = int(rows.min()), int(rows.max())
        dist_m = abs(
            GEOD.inv(
                region_lons[r0, c],
                region_lats[r0, c],
                region_lons[r1, c],
                region_lats[r1, c],
            )[2]
        )
        height_km = dist_m / 1000.0

    return height_km, width_km
