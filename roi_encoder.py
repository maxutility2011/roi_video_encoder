import argparse
import os
import sys
from fractions import Fraction

import av
import cv2
import numpy as np
from inference.models.utils import get_roboflow_model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "native"))
try:
    import roi_sidedata
except ImportError as e:
    raise SystemExit(
        "native/roi_sidedata.so not found or failed to load "
        f"({e}). Build it first with: native/build.sh"
    )

# ---------------------------------------------------------------------------
# ROI detection: YOLO on each decoded frame, every matching detection becomes
# its own ROI (all boosted together, sharing one --qoffset).
# ---------------------------------------------------------------------------
# yolo26n-640 is a closed-set COCO detector (80 fixed classes: "person", "dog",
# "car", "cat", ... - see https://cocodataset.org/#explore for the full list),
# not open-vocabulary, so --roi must name one or more of those class strings.
# There's no "face" class in COCO; see get_box_face() below for an actual face ROI.
MODEL_ID = "yolo26n-640"
DEFAULT_ROI_WORDS = ["person"]
DEFAULT_CONFIDENCE = 0.4
DEFAULT_CRF = 43
DEFAULT_QOFFSET = "-0.5"

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = get_roboflow_model(MODEL_ID)
    return _model


def get_boxes(frame, roi_words=DEFAULT_ROI_WORDS, confidence=DEFAULT_CONFIDENCE):
    """Detect all ROI instances in one RGB frame and return their boxes.

    roi_words is a list of COCO class names that count as the ROI, e.g.
    ["person"] or ["person", "dog"].

    Returns a list of (x, y, w, h) as integer pixels, one per detection:
    (x, y) is the top-left corner, (w, h) the size. Returns an empty list
    when nothing is detected.
    """
    # inference/OpenCV work in BGR; our frames are RGB.
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    result = _get_model().infer(
        bgr,
        confidence=confidence,
        class_filter=roi_words,
    )[0]

    fh, fw = frame.shape[:2]
    boxes = []
    for pred in result.predictions:
        # inference returns center-x/center-y plus width/height.
        x = int(round(pred.x - pred.width / 2))
        y = int(round(pred.y - pred.height / 2))
        w = int(round(pred.width))
        h = int(round(pred.height))

        # clamp to the frame - side data with an out-of-bounds region is rejected.
        x = max(0, min(x, fw - 1))
        y = max(0, min(y, fh - 1))
        w = max(0, min(w, fw - x))
        h = max(0, min(h, fh - y))
        if w > 0 and h > 0:
            boxes.append((x, y, w, h))
    return boxes


# ---- Alternative: real face ROI using OpenCV's bundled Haar cascade --------
# Fully offline, no model download. Swap get_box -> get_box_face in the loop
# below if the ROI you want is faces rather than whole people.
_face_cascade = None


def get_box_face(frame):
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                           minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return 0, 0, 0, 0
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(w), int(h)


# ---------------------------------------------------------------------------
# Driver: single-pass PyAV decode -> detect -> attach ROI side data -> encode
# ---------------------------------------------------------------------------
# This bypasses ffmpeg's CLI/filter graph entirely. It exists because
# ffmpeg's addroi filter cannot vary its region per frame through any CLI
# mechanism in this build: its x/y/w/h options are evaluated once at
# filter-graph init (not per frame, confirmed by testing a per-frame
# expression, which fails outright), and aren't marked runtime-tunable for
# sendcmd either (confirmed via `ffmpeg -h filter=addroi`). Attaching the
# side data directly to the frame object that was actually detected on means
# it can never desync from that frame - no scheduling, no timestamps.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-encode a video, giving extra bits (lower QP) to a detected ROI.",
    )
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument(
        "-o", "--output", default="output.mp4",
        help="Path to write the ROI-encoded output video (default: %(default)s)",
    )
    parser.add_argument(
        "--roi", default=",".join(DEFAULT_ROI_WORDS),
        help=(
            "Comma-separated COCO class name(s) to treat as the ROI, e.g. 'person' "
            f"or 'person,dog,car'. Must be class(es) the {MODEL_ID} model knows - "
            "it's a fixed-vocabulary COCO detector, not open-vocabulary. Every "
            "matching detection in a frame is boosted simultaneously (e.g. "
            "'person,car' boosts every detected person AND every detected car "
            "in the same frame, each getting the same --qoffset), not just "
            "the single largest match. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--conf", type=float, default=DEFAULT_CONFIDENCE,
        help="Confidence threshold for the ROI detector (default: %(default)s)",
    )
    parser.add_argument(
        "--crf", type=int, default=DEFAULT_CRF,
        help=(
            "libx264 Constant Rate Factor for the overall encode - lower is "
            "higher quality/bitrate, higher is lower quality/bitrate, range "
            "0-51 (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--qoffset", default=DEFAULT_QOFFSET,
        help=(
            "Quantizer offset applied inside the detected ROI, as a number in "
            "[-1, 1], e.g. '-0.5' or the equivalent rational '-1/2'. Negative "
            "values lower the ROI's QP (more bits/better quality inside the "
            "ROI relative to the rest of the frame). Since it starts with "
            "'-', write it as --qoffset=-0.5 (with '=') or argparse will "
            "mistake it for another flag. (default: %(default)s)"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    roi_words = [w.strip() for w in args.roi.split(",") if w.strip()]
    if not roi_words:
        raise SystemExit("--roi must name at least one COCO class")

    qoffset = Fraction(args.qoffset).limit_denominator(1000)

    # Load the detector eagerly, with its own status line - on a first run this
    # downloads model weights, which can take a while and would otherwise look
    # exactly like a hang once the frame loop starts.
    print(f"Loading ROI detector model: {MODEL_ID} ...", flush=True)
    _get_model()
    print("Model ready.", flush=True)

    in_container = av.open(args.video)
    in_stream = in_container.streams.video[0]
    total_frames = in_stream.frames  # container metadata; can be 0/unreliable
    print(
        f"Input: {args.video} ({in_stream.width}x{in_stream.height} "
        f"@ {float(in_stream.average_rate):g}fps"
        + (f", ~{total_frames} frames)" if total_frames > 0 else ", frame count unknown)"),
        flush=True,
    )

    out_container = av.open(args.output, mode="w")
    out_stream = out_container.add_stream("libx264", rate=in_stream.average_rate)
    out_stream.width = in_stream.width
    out_stream.height = in_stream.height
    out_stream.pix_fmt = "yuv420p"
    out_stream.time_base = in_stream.time_base
    out_stream.options = {"crf": str(args.crf)}

    print("Encoding with per-frame ROI applied...", flush=True)
    frame_idx = 0
    for frame in in_container.decode(in_stream):
        # Detect on an RGB view of the frame - this doesn't touch frame.ptr,
        # so it's independent of whatever reformatting happens below.
        rgb = frame.to_ndarray(format="rgb24")
        boxes = get_boxes(rgb, roi_words=roi_words, confidence=args.conf)

        # Reformat to the encoder's pixel format BEFORE attaching ROI side
        # data: reformat() builds a new underlying AVFrame, and side data
        # attached to the pre-reformat frame is not guaranteed to carry over.
        if frame.format.name != out_stream.pix_fmt:
            frame = frame.reformat(format=out_stream.pix_fmt)

        if boxes:
            # IMPORTANT: attach before anything ever reads frame.side_data -
            # PyAV's side-data view is built and cached on first access, so
            # a read beforehand (even just to log) would hide this from any
            # later Python-level check.
            regions = [(y, y + h, x, x + w) for x, y, w, h in boxes]
            roi_sidedata.attach_rois(
                frame, regions,
                qnum=qoffset.numerator, qden=qoffset.denominator,
            )

        for packet in out_stream.encode(frame):
            out_container.mux(packet)

        frame_idx += 1
        if total_frames > 0:
            pct = frame_idx / total_frames * 100
            print(f"\rFrame {frame_idx}/{total_frames} ({pct:5.1f}%)", end="", flush=True)
        else:
            print(f"\rFrame {frame_idx}", end="", flush=True)

    for packet in out_stream.encode():  # flush the encoder
        out_container.mux(packet)

    out_container.close()
    in_container.close()
    print(f"\nDone: {frame_idx} frame(s) processed -> {args.output}")


if __name__ == "__main__":
    main()
