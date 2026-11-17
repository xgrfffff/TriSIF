# Meteosat–SIRTA Alignment

Build aligned sky-image and satellite sequences from preprocessed SIRTA JPGs and Meteosat SEVIRI Native files.

## Files

| File | Purpose |
| --- | --- |
| `build_alignment_sequences.py` | Match timestamps, crop satellite images, and save aligned sequences. |
| `eumetsat_alignment_utils.py` | Parse filenames, read sky images, and define station and channel settings. |
| `requirements.txt` | Python dependencies. |

## Setup

Use Python 3.10 or later. Run the following commands from this directory:

```bash
python -m pip install -r requirements.txt
```

## Input

Place satellite files directly under the input root and sky images in date folders:

```text
input_root/
├── MSG4-SEVI-MSG15-0100-NA-20230701080000.000000000Z-NA.nat
├── ...
└── 20230701/
    ├── 20230701080000.jpg
    ├── 20230701081000.jpg
    └── ...
```

- Sky filenames must start with `YYYYMMDDhhmmss`. Images must already be corrected/cropped as needed and have consistent dimensions.
- Native filenames must match `MSG4-SEVI-MSG15-0100-NA-YYYYMMDD*.nat` and contain `-YYYYMMDDhhmmss.<digits>Z-`. Both `IR_108` and `HRV` channels are required.
- Both filename timestamps must use **UTC**. Sky timestamps must match the requested times exactly; satellite filename timestamps must fall within the following 5-minute slot by default.

## Run

Replace the input path and dates with your own:

```bash
python build_alignment_sequences.py \
  --data-root "/path/to/input_root" \
  --dates 20230701 20230702 \
  --window-minutes 120 \
  --sky-step-minutes 10 \
  --satellite-step-minutes 10 \
  --window-stride-minutes 10 \
  --output-dir "outputs/aligned_sequences"
```

This creates 120-minute windows with 12 sky frames and 12 satellite frames per sample. Windows with missing frames are excluded. Each NPZ contains RGB sky images, IR/HRV crops, timestamps, and source filenames. A summary is saved to `outputs/alignment_summary.json`.

Use `python build_alignment_sequences.py --help` for additional options. This step does not download data or generate irradiance labels or sun masks.
