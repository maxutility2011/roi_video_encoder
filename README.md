# roi_video_encoder

Re-encodes a video, giving extra bits (lower QP) to a detected region of
interest (e.g. a person) via a per-frame ROI attached directly to each
frame before `libx264` encoding.

## Setup

Requires Python 3, `gcc`, and `curl` (the last two only for the one-time
native build below).

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Build the native extension (only needed once per machine, or again after
upgrading the `av` package - `native/build.sh` downloads matching FFmpeg
headers for whatever `av` version is installed, no `sudo` required):

```bash
native/build.sh
```

## Run

```bash
python3 roi_encoder.py <input_video> -o <output_video> --roi person --conf 0.4 --crf 40 --qoffset=-0.5
```

- `<input_video>` - path to the input video file (required, positional).
- `-o, --output` - path to write the output video (default: `output.mp4`).
- `--roi` - comma-separated COCO class name(s) to treat as the ROI, e.g.
  `person` or `person,dog,car` (default: `person`).
- `--conf` - detector confidence threshold (default: `0.4`).
- `--crf` - libx264 Constant Rate Factor for the overall encode; lower is
  higher quality/bitrate (default: `43`).
- `--qoffset` - QP offset applied inside the ROI, e.g. `-0.5` or `-1/2`;
  negative values improve quality inside the ROI. Use `--qoffset=<value>`
  (with `=`) since a leading `-` is otherwise misread as another flag.

Example:

```bash
python3 roi_encoder.py cooking.mp4 -o out.mp4 --roi person --conf 0.4 --crf 40 --qoffset=-0.5
```
