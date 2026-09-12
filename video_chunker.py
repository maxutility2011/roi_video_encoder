import argparse
import os

import av
import cv2
from av.video.frame import PictureType

# ---------------------------------------------------------------------------
# Scene-cut detection: per-frame color histogram correlation against the
# previous frame, on a heavily downscaled grayscale copy. This is a cheap,
# dependency-free stand-in for a real shot-boundary detector (e.g.
# PySceneDetect) - good enough to pick chunk boundaries, not a general-purpose
# scene classifier.
# ---------------------------------------------------------------------------
SCENE_DETECT_SIZE = 32  # downscale to SIZE x SIZE before histogramming
SCENE_CUT_CORREL_THRESHOLD = 0.6  # correlation below this counts as a cut

MAX_DEVIATION_PCT = 45.0  # must stay well under 50 or adjacent search windows overlap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a video into chunks along scene-cut boundaries, so parallel "
            "per-chunk ROI encoding doesn't break a single scene across chunks."
        ),
    )
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument(
        "-n", "--chunks", type=int, required=True,
        help="Number of output chunks to split the video into",
    )
    parser.add_argument(
        "-d", "--deviation", type=float, default=10.0,
        help=(
            "How far each chunk boundary may move from its nominal position "
            "(input duration / --chunks), as a percentage of the nominal chunk "
            "duration, e.g. 10 means +/-10%%. Within that window the boundary "
            "snaps to the nearest detected scene cut; if none falls inside the "
            "window, it falls back to the nominal (hard-cut) boundary. "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "-o", "--output-dir", default="chunks",
        help="Directory to write chunk_0000.mp4, chunk_0001.mp4, ... into (default: %(default)s)",
    )
    args = parser.parse_args()
    if args.chunks < 1:
        raise SystemExit("--chunks must be >= 1")
    if not (0 <= args.deviation < MAX_DEVIATION_PCT):
        raise SystemExit(f"--deviation must be in [0, {MAX_DEVIATION_PCT}) percent")
    return args


def analyze_video(video_path, correl_threshold=SCENE_CUT_CORREL_THRESHOLD):
    """Single decode pass over the whole video, returning:
      - duration_sec: timestamp of the last decoded frame, in seconds
      - scene_cuts: sorted list of timestamps (seconds) flagged as scene cuts
      - keyframe_times: sorted list of every I-frame's timestamp (seconds)

    keyframe_times matters because stream-copy splitting (no re-encode) can
    only start a new chunk on a frame the original stream already made an
    I-frame - that's the only point a decoder can start from cold.
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    time_base = stream.time_base

    duration_sec = 0.0
    scene_cuts = []
    keyframe_times = []
    prev_hist = None

    for frame in container.decode(stream):
        if frame.pts is None:
            continue
        t = float(frame.pts * time_base)
        duration_sec = max(duration_sec, t)

        if frame.pict_type == PictureType.I:
            keyframe_times.append(t)

        gray = cv2.cvtColor(frame.to_ndarray(format="rgb24"), cv2.COLOR_RGB2GRAY)
        small = cv2.resize(gray, (SCENE_DETECT_SIZE, SCENE_DETECT_SIZE))
        hist = cv2.calcHist([small], [0], None, [32], [0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        if prev_hist is not None:
            similarity = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if similarity < correl_threshold:
                scene_cuts.append(t)
        prev_hist = hist

    container.close()
    return duration_sec, scene_cuts, keyframe_times


def pick_boundaries(duration_sec, num_chunks, deviation_pct, scene_cuts, keyframe_times):
    """Choose up to num_chunks - 1 interior cut points, each expressed in
    seconds from the start of the video.

    For each nominal boundary (i * duration/num_chunks), prefer the nearest
    scene cut within +/- deviation_pct% of the nominal chunk duration; fall
    back to the nominal point itself if no scene cut falls in that window.
    Either way, the chosen point is then snapped backward to the nearest
    keyframe at or before it, since stream-copy splitting requires every
    chunk to start on one. A boundary is dropped (producing a shorter chunk
    list than requested) if no keyframe is available after the previous
    boundary and at or before the chosen point.
    """
    if num_chunks <= 1:
        return []

    nominal_duration = duration_sec / num_chunks
    window = nominal_duration * (deviation_pct / 100.0)
    scene_cut_times = sorted(scene_cuts)
    keyframe_times = sorted(keyframe_times)

    boundaries = []
    prev_boundary = 0.0
    for i in range(1, num_chunks):
        nominal = i * nominal_duration
        lo, hi = nominal - window, nominal + window

        candidates = [t for t in scene_cut_times if lo <= t <= hi and t > prev_boundary]
        chosen = min(candidates, key=lambda t: abs(t - nominal)) if candidates else nominal

        eligible_kf = [t for t in keyframe_times if prev_boundary < t <= chosen]
        if not eligible_kf:
            print(
                f"Warning: no keyframe available for boundary ~{nominal:.2f}s "
                f"(chosen cut {chosen:.2f}s) - skipping this boundary.",
                flush=True,
            )
            continue

        snapped = max(eligible_kf)
        boundaries.append(snapped)
        prev_boundary = snapped

    return boundaries


def split_video(video_path, boundaries, output_dir):
    """Stream-copy (no re-encode) the input into len(boundaries) + 1 chunk
    files at the given cut points, each chunk's timestamps rebased to start
    at zero.
    """
    os.makedirs(output_dir, exist_ok=True)
    in_container = av.open(video_path)
    in_stream = in_container.streams.video[0]
    time_base = in_stream.time_base

    cut_points = list(boundaries) + [float("inf")]
    chunk_idx = 0
    out_container = None
    out_stream = None
    pts_offset = None
    chunk_paths = []

    def open_chunk():
        nonlocal out_container, out_stream, pts_offset
        path = os.path.join(output_dir, f"chunk_{chunk_idx:04d}.mp4")
        out_container = av.open(path, mode="w")
        out_stream = out_container.add_stream_from_template(in_stream)
        out_stream.time_base = in_stream.time_base
        pts_offset = None
        chunk_paths.append(path)
        print(f"Writing {path} ...", flush=True)

    open_chunk()

    for packet in in_container.demux(in_stream):
        if packet.pts is None:
            continue  # trailing flush packet from the demuxer; nothing to place
        t = float(packet.pts * time_base)

        if packet.is_keyframe and t >= cut_points[chunk_idx]:
            out_container.close()
            chunk_idx += 1
            open_chunk()

        if pts_offset is None:
            pts_offset = packet.pts

        packet.pts -= pts_offset
        packet.dts -= pts_offset
        packet.stream = out_stream
        out_container.mux(packet)

    out_container.close()
    in_container.close()
    return chunk_paths


def main():
    args = parse_args()

    print(f"Analyzing {args.video} for scene cuts and keyframes...", flush=True)
    duration_sec, scene_cuts, keyframe_times = analyze_video(args.video)
    print(
        f"Duration ~{duration_sec:.2f}s, {len(scene_cuts)} candidate scene cut(s), "
        f"{len(keyframe_times)} keyframe(s) found.",
        flush=True,
    )

    boundaries = pick_boundaries(duration_sec, args.chunks, args.deviation, scene_cuts, keyframe_times)
    if len(boundaries) < args.chunks - 1:
        print(
            f"Requested {args.chunks} chunks but only {len(boundaries)} usable "
            f"boundary(ies) found; producing {len(boundaries) + 1} chunk(s) instead.",
            flush=True,
        )

    chunk_paths = split_video(args.video, boundaries, args.output_dir)
    print(f"Done: wrote {len(chunk_paths)} chunk(s) to {args.output_dir}/", flush=True)


if __name__ == "__main__":
    main()
