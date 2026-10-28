"""UVQ 1.5 inference with a TensorRT engine.

Drop-in replacement for `uvq_inference.py --model_version 1.5 --device cuda`
from google/uvq: same frame sampling, same scores (up to FP16 rounding when
the engine was built with --fp16) and the same output format, but the three
UVQ networks run as one TensorRT engine and FFmpeg decoding overlaps with
inference.

Examples:
    python3 uvq_trt/uvq_trt_inference.py video.mp4 --engine uvq_trt/uvq1p5_fp16.engine
    python3 uvq_trt/uvq_trt_inference.py video.mp4 --engine uvq_trt/uvq1p5_fp16.engine --output_all_stats
    python3 uvq_trt/uvq_trt_inference.py videos.txt --engine uvq_trt/uvq1p5_fp16.engine --output scores.txt
"""

import argparse
import json
import math
import os
import queue
import sys
import threading
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import uvq1p5_trt  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", help="A video file, or a .txt file with one video path per line")
    parser.add_argument("--engine", required=True, help="Engine built by build_engine.py")
    parser.add_argument("--uvq_dir", default=uvq1p5_trt.DEFAULT_UVQ_DIR, help="Path to a google/uvq checkout")
    parser.add_argument("--output", default="", help="JSON output (single video) or CSV lines (.txt input)")
    parser.add_argument("--transpose", action="store_true", help="Transpose the video before processing")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second to sample; -1 for all frames")
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Frames decoded per batch (default: the engine's max batch)"
    )
    parser.add_argument("--hwaccel", default=None, help="FFmpeg -hwaccel for decoding, e.g. cuda")
    parser.add_argument("--output_all_stats", action="store_true", help="Print all stats in JSON format")
    parser.add_argument(
        "--compare_pytorch", action="store_true", help="Also run the original PyTorch UVQ on CUDA and report the diff"
    )
    parser.add_argument("--ffmpeg_path", default="ffmpeg")
    parser.add_argument("--ffprobe_path", default="ffprobe")
    return parser.parse_args()


def prefetch(iterator, depth=2):
    """Runs a generator in a background thread so decoding overlaps the GPU."""
    q = queue.Queue(maxsize=depth)
    done = object()

    def worker():
        try:
            for item in iterator:
                q.put(item)
        except BaseException as e:  # re-raised in the consumer
            q.put(e)
        q.put(done)

    threading.Thread(target=worker, daemon=True).start()
    while (item := q.get()) is not done:
        if isinstance(item, BaseException):
            raise item
        yield item


def video_timing(path, fps, ffprobe_path):
    """Returns (video_length_s, fps_to_use, orig_fps) following uvq_inference.py."""
    import probe  # from the UVQ checkout (utils/probe.py)

    duration = probe.get_video_duration(path, ffprobe_path)
    orig_fps = probe.get_r_frame_rate(path, ffprobe_path)
    nb_frames = probe.get_nb_frames(path, ffprobe_path)

    video_length = 0
    if duration:
        video_length = math.ceil(duration)
    elif orig_fps and nb_frames:
        video_length = math.ceil(nb_frames / orig_fps)

    if fps == -1:
        if not orig_fps:
            raise RuntimeError(f"Cannot determine r_frame_rate for {path} with --fps -1")
        fps = orig_fps
        if nb_frames:
            video_length = math.ceil(nb_frames / fps)
    if video_length == 0:
        raise RuntimeError(f"Could not determine video length for {path}")
    return video_length, fps, orig_fps


def score_video(engine, path, args, transpose, reference=None):
    video_length, fps, orig_fps = video_timing(path, args.fps, args.ffprobe_path)
    batches = uvq1p5_trt.iter_uint8_batches(
        path,
        video_length,
        transpose,
        video_fps=fps,
        batch_size=args.batch_size,
        ffmpeg_path=args.ffmpeg_path,
        hwaccel=args.hwaccel,
    )

    predictions, reference_predictions = [], []
    with torch.inference_mode():
        for frames, _ in prefetch(batches):
            gpu_frames = torch.from_numpy(frames).to("cuda")
            predictions.append(engine(gpu_frames))
            if reference is not None:
                video = gpu_frames.permute(0, 3, 1, 2).float() * (2.0 / 255.0) - 1.0
                reference_predictions.append(reference.uvq1p5_core(video.unsqueeze(1)))

    prediction = torch.cat(predictions)
    frame_scores = prediction.cpu().flatten().tolist()
    if orig_fps:
        frame_indices = [int(round(i * orig_fps / fps)) for i in range(len(frame_scores))]
    else:
        frame_indices = list(range(len(frame_scores)))
    results = {
        "uvq1p5_score": prediction.mean().item(),
        "per_frame_scores": frame_scores,
        "frame_indices": frame_indices,
    }
    if reference is not None:
        ref = torch.cat(reference_predictions)
        results["pytorch_uvq1p5_score"] = ref.mean().item()
        results["max_abs_frame_diff"] = (prediction - ref).abs().max().item()
    return results


def is_portrait(path, ffprobe_path):
    import probe

    dimensions = probe.get_dimensions(path, ffprobe_path)
    return bool(dimensions and dimensions[0] < dimensions[1])


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("TensorRT inference needs a CUDA GPU")
    uvq1p5_trt.add_uvq_to_path(args.uvq_dir)

    engine = uvq1p5_trt.TRTEngine(args.engine)
    if args.batch_size is None:
        args.batch_size = engine.max_batch
    if args.batch_size < 1:
        raise SystemExit("--batch_size must be at least 1")
    reference = uvq1p5_trt.load_uvq1p5(args.uvq_dir).cuda() if args.compare_pytorch else None

    if args.input.endswith(".txt"):
        if not args.output:
            raise SystemExit("--output must be specified when input is a .txt file list")
        with open(args.input) as f:
            paths = [line.strip() for line in f if line.strip()]
        lines = []
        start = time.time()
        for path in paths:
            try:
                transpose = is_portrait(path, args.ffprobe_path)
                score = score_video(engine, path, args, transpose, reference)["uvq1p5_score"]
                lines.append(f"{os.path.basename(path)},{score}")
            except Exception as e:
                print(f"Error processing {path}: {e}", file=sys.stderr)
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            f.writelines(line + "\n" for line in lines)
        print(f"Scored {len(lines)}/{len(paths)} videos in {time.time() - start:.1f}s -> {args.output}")
        return

    start = time.time()
    results = score_video(engine, args.input, args, args.transpose, reference)
    results["video_name"] = args.input
    print(json.dumps(results) if args.output_all_stats else results["uvq1p5_score"])
    print(f"({time.time() - start:.2f}s)", file=sys.stderr)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
