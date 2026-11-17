# GOES–SRRL Alignment

Build aligned sky-image and satellite sequences from preprocessed SRRL dual-exposure JPGs and GOES ABI L1b NetCDF files.

## Files

| File | Purpose |
| --- | --- |
| `build_alignment_sequences.py` | Command-line entry point and parameter validation. |
| `alignment.py` | Build consecutive sequences, match satellite observations, and save NPZ files. |
| `data_io.py` | Read images, parse timestamps, and locate/crop GOES pixels around SRRL. |
| `requirements.txt` | Python dependencies. |

## Setup

Use Python 3.10 or later. Run the following commands from this directory:

```bash
python -m pip install -r requirements.txt
```

## Input

```text
input_root/
├── srrl_aberration/
│   └── 20210101/
│       ├── 20210101080000_11.jpg
│       ├── 20210101080000_12.jpg
│       └── ...
├── srrl_cut/
│   └── 20210101/
│       ├── 20210101080000_11.jpg
│       ├── 20210101080000_12.jpg
│       └── ...
└── goes_C13/
    └── 2021/
        └── 1/                 # UTC day of year; 001 is also supported
            └── *.nc
```

- Each SRRL timestamp requires both `_11` and `_12` exposures in both the corrected (`srrl_aberration`) and cropped (`srrl_cut`) folders. Image dimensions must be consistent within each stream.
- SRRL filenames use `YYYYMMDDhhmmss` in fixed **MST (UTC−7)**, without daylight saving time.
- SRRL images must already be sampled at the requested interval. The builder does not subsample one-minute images into ten-minute frames.
- Place GOES files directly in the day-of-year folders, retaining their standard `_s..._e...` filenames. Use one channel/platform/product/grid per input root, such as GOES-16 ABI L1b RadC C13.

## Run

Replace the input paths and year with your own:

```bash
python build_alignment_sequences.py \
  --srrl-aberration-root "/path/to/input_root/srrl_aberration" \
  --srrl-cut-root "/path/to/input_root/srrl_cut" \
  --goes-year-root "/path/to/input_root/goes_C13/2021" \
  --year 2021 \
  --sequence-length 12 \
  --srrl-interval-minutes 10 \
  --goes-interval-minutes 5 \
  --workers 1 \
  --output-dir "outputs/aligned_sequences"
```

With these settings, each complete NPZ contains 12 frames for each SRRL image stream, 24 GOES crops of size `128×128`, timestamps, and crop coordinates. Incomplete sequences are skipped. Check terminal messages and `outputs/aligned_sequences/build_alignment_sequences.log` for results.

GOES matching retains the original fixed coverage interval `[start_time, start_time + 5 minutes)`. Changing `--goes-interval-minutes` changes target sampling only; it does not change coverage. With 12 SRRL frames at ten-minute intervals, setting the GOES interval to ten minutes produces 13 GOES targets under the original endpoint rule.

Use `python build_alignment_sequences.py --help` for additional options. This step does not download data, correct/crop SRRL images, or generate irradiance labels or sun masks.
