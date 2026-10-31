# UVQ 1.5 on TensorRT

Scripts to serialize Google's [UVQ](https://github.com/google/uvq) 1.5
no-reference video quality model into a TensorRT engine and run inference
with it, e.g. to score the output of `roi_encoder.py`.

UVQ 1.5 is three PyTorch networks: ContentNet, DistortionNet (two
EfficientNet-B0 backbones) and AggregationNet. Rather than three engines,
`export_onnx.py` fuses all of them, including the input normalization, the
3x3 patch split of DistortionNet and ContentNet's 256x256 resize, into **one
ONNX graph -> one TensorRT engine**:

| tensor   | shape                    | dtype   |                                  |
|----------|--------------------------|---------|----------------------------------|
| `frames` | `(batch, 1080, 1920, 3)` | uint8   | RGB, as FFmpeg's `rgb24` output   |
| `scores` | `(batch, 1)`             | float32 | per-frame UVQ 1.5 score in [1, 5] |

Only the batch dimension is dynamic (UVQ always rescales frames to 1920x1080).
Feeding uint8 instead of float32 also cuts host->GPU traffic by 4x.

Only UVQ 1.5 (UVQ's default and recommended version) is covered, not UVQ 1.0.

The same exported ONNX graph can also run on the CPU with ONNX Runtime
(`--onnx` instead of `--engine`), no GPU or TensorRT needed - useful to check
the conversion on your own videos before building an engine.

## Requirements

- Python 3.10-3.13, `pip install -r uvq_trt/requirements.txt`
- `ffmpeg` / `ffprobe` on `PATH`
- For the TensorRT path only: an NVIDIA GPU, CUDA, and TensorRT >= 8.6
  (TensorRT 10 recommended) with its Python bindings, e.g.
  `pip install tensorrt`

Get the UVQ model code and checkpoints (cloned to `uvq_trt/uvq/`, pinned to a
known revision):

```bash
uvq_trt/fetch_uvq.sh
```

On Windows, see [Windows, CPU only](#windows-cpu-only) below instead.

## 1. Export ONNX (any machine, CPU is fine)

```bash
python3 uvq_trt/export_onnx.py -o uvq_trt/uvq1p5.onnx
```

The export is checked with ONNX Runtime against the unmodified PyTorch UVQ
model and fails if the scores differ by more than 1e-3.

## 2. Build the TensorRT engine (on the GPU machine that will run it)

Engines are specific to the GPU model and TensorRT version, so build on the
inference machine:

```bash
python3 uvq_trt/build_engine.py uvq_trt/uvq1p5.onnx -o uvq_trt/uvq1p5_fp16.engine --fp16
```

- `--fp16` - use FP16 kernels; the main speedup on Tensor Core GPUs. Omit for
  an FP32 engine (the scores then match PyTorch to ~1e-5).
- `--min_batch/--opt_batch/--max_batch` - engine batch range (default
  1/24/24 frames, matching UVQ's CUDA batch size). Larger batches need more
  GPU memory: each frame is run as 9 DistortionNet patches.
- `--timing_cache FILE` - reuse kernel timings to make rebuilds faster.

Or with `trtexec`:

```bash
trtexec --onnx=uvq_trt/uvq1p5.onnx --saveEngine=uvq_trt/uvq1p5_fp16.engine --fp16 --minShapes=frames:1x1080x1920x3 --optShapes=frames:24x1080x1920x3 --maxShapes=frames:24x1080x1920x3
```

## 3. Run inference

Same arguments and output as UVQ's `uvq_inference.py --model_version 1.5`:

```bash
python3 uvq_trt/uvq_trt_inference.py video.mp4 --engine uvq_trt/uvq1p5_fp16.engine
```

```bash
python3 uvq_trt/uvq_trt_inference.py video.mp4 --engine uvq_trt/uvq1p5_fp16.engine --output_all_stats
```

```bash
python3 uvq_trt/uvq_trt_inference.py videos.txt --engine uvq_trt/uvq1p5_fp16.engine --output scores.txt
```

- `--fps` - frames sampled per second (default `1`, `-1` = every frame).
- `--hwaccel cuda` - decode with NVDEC. With TensorRT, FFmpeg decoding +
  bicubic rescale to 1080p is usually the bottleneck, and the decoder is
  bit-exact, so this speeds things up without changing the scores.
- `--compare_pytorch` - also run the original PyTorch UVQ on the same device and
  report `pytorch_uvq1p5_score` and `max_abs_frame_diff`; use it once to
  check an FP16 engine's accuracy on your content.
- `--transpose`, `--batch_size`, `--output`, `--ffmpeg_path`,
  `--ffprobe_path` - as in UVQ.

Decoding runs in a background thread, so it overlaps with GPU inference.

## Running on the CPU with ONNX Runtime

Pass the ONNX file from step 1 with `--onnx` instead of an engine; no GPU,
CUDA or TensorRT needed, and step 2 is skipped:

```bash
python3 uvq_trt/uvq_trt_inference.py video.mp4 --onnx uvq_trt/uvq1p5.onnx --compare_pytorch --output_all_stats
```

`--threads N` limits the CPU threads; all other options work as above.
With `--compare_pytorch` the output includes `pytorch_uvq1p5_score` (the
original UVQ model on the same frames) and `max_abs_frame_diff`, which
should be around 1e-6.

For example, UVQ's sample clip
[Gaming_1080P-0ce6_orig.mp4](https://storage.googleapis.com/ugc-dataset/vp9_compressed_videos/Gaming_1080P-0ce6_orig.mp4)
scores `3.8483` both this way and with UVQ's own `uvq_inference.py` at the
pinned revision (UVQ's README still lists an older `3.8804`). Small
differences are expected with other FFmpeg versions, since FFmpeg does the
frame sampling and rescaling.

## Windows, CPU only

Needs Python 3.10-3.13 (not 3.14: there are no Windows wheels of
`torch==2.8.0` for it), Git and FFmpeg, e.g. with
`winget install Python.Python.3.13`, `winget install Git.Git` and
`winget install Gyan.FFmpeg`, then reopen PowerShell. In PowerShell, from the
root of this repository:

```powershell
git clone https://github.com/google/uvq.git uvq_trt\uvq
git -C uvq_trt\uvq checkout c204ecc04f093e90c941cb126c7ac3d5beca0414
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r uvq_trt\requirements.txt
python uvq_trt\export_onnx.py -o uvq_trt\uvq1p5.onnx
python uvq_trt\uvq_trt_inference.py video.mp4 --onnx uvq_trt\uvq1p5.onnx --compare_pytorch --output_all_stats
```

If `Activate.ps1` is blocked ("running scripts is disabled"), run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.
