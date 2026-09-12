import argparse
import os
import sys
import time
from fractions import Fraction

print("roi_encoder: GitHub push connectivity test")

import av
import cv2
import numpy as np
from av.codec.context import Flags2
from av.sidedata.sidedata import Type as SideDataType
from av.video.frame import PictureType
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


# ---------------------------------------------------------------------------
# Text detection: OpenCV's EAST detector, selected via the reserved --roi
# word "text" (mixable with COCO classes, e.g. --roi person,text).
# ---------------------------------------------------------------------------
TEXT_ROI_WORD = "text"
EAST_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "frozen_east_text_detection.pb")
EAST_INPUT_SIZE = 320  # must be a multiple of 32; smaller = faster, less accurate on small text
EAST_NMS_THRESHOLD = 0.4

_east_net = None


def _get_east_net():
    global _east_net
    if _east_net is None:
        if not os.path.isfile(EAST_MODEL_PATH):
            raise SystemExit(
                f"EAST text detection model not found at {EAST_MODEL_PATH}. "
                "Download it first with: models/download_east.sh"
            )
        _east_net = cv2.dnn.readNet(EAST_MODEL_PATH)
    return _east_net


def get_text_boxes(frame, confidence=DEFAULT_CONFIDENCE, nms_threshold=EAST_NMS_THRESHOLD):
    """Detect text regions in one RGB frame using OpenCV's EAST detector.

    Returns a list of (x, y, w, h) axis-aligned boxes, one per detected
    text region. EAST natively predicts rotated boxes; we approximate each
    as axis-aligned (ignoring the small rotation typical of on-screen text/
    captions) since AVRegionOfInterest only supports axis-aligned rectangles
    anyway - the same simplification used in most EAST tutorials.
    """
    net = _get_east_net()
    fh, fw = frame.shape[:2]
    scale_x = fw / EAST_INPUT_SIZE
    scale_y = fh / EAST_INPUT_SIZE

    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    blob = cv2.dnn.blobFromImage(
        bgr, 1.0, (EAST_INPUT_SIZE, EAST_INPUT_SIZE),
        (123.68, 116.78, 103.94), swapRB=True, crop=False,
    )
    net.setInput(blob)
    scores, geometry = net.forward([
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3",
    ])

    rects = []
    confidences = []
    num_rows, num_cols = scores.shape[2:4]
    for row in range(num_rows):
        scores_row = scores[0, 0, row]
        x0, x1, x2, x3 = geometry[0, 0, row], geometry[0, 1, row], geometry[0, 2, row], geometry[0, 3, row]
        angles_row = geometry[0, 4, row]
        for col in range(num_cols):
            if scores_row[col] < confidence:
                continue
            offset_x, offset_y = col * 4.0, row * 4.0
            angle = angles_row[col]
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            box_h = x0[col] + x2[col]
            box_w = x1[col] + x3[col]
            end_x = offset_x + cos_a * x1[col] + sin_a * x2[col]
            end_y = offset_y - sin_a * x1[col] + cos_a * x2[col]
            start_x = end_x - box_w
            start_y = end_y - box_h
            rects.append((start_x, start_y, box_w, box_h))
            confidences.append(float(scores_row[col]))

    if not rects:
        return []

    indices = cv2.dnn.NMSBoxes(rects, confidences, confidence, nms_threshold)
    boxes = []
    for i in np.array(indices).flatten():
        x, y, w, h = rects[i]
        x = int(round(x * scale_x))
        y = int(round(y * scale_y))
        w = int(round(w * scale_x))
        h = int(round(h * scale_y))

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
# Compressed-domain box propagation: rather than running the (expensive)
# detectors on every frame, we only detect on I-frames and carry the boxes
# forward on P/B-frames using the motion vectors FFmpeg's decoder already
# computed during motion compensation - no extra decode cost, no detector
# call. This trades a bit of tracking accuracy (translate-only, no
# scaling/rotation, and a same-frame average when a box straddles several
# motion vectors moving differently) for skipping the detector on most
# frames.
# ---------------------------------------------------------------------------
def propagate_boxes(boxes, motion_vectors, frame_width, frame_height):
    """Shift each box in `boxes` using nearby motion vectors from the frame
    that followed it, instead of re-running detection on that frame.

    Each AVMotionVector records a block move from (src_x, src_y) in the
    reference frame to (dst_x, dst_y) in this frame. For a box positioned in
    the reference frame's coordinates, we average the displacement
    (dst - src) of every motion vector whose source block center falls
    inside that box, then translate the box by that average. A box with no
    motion vectors landing inside it (e.g. an all-intra region, or no motion
    vector data at all) is passed through unchanged, i.e. assumed static.
    """
    if not boxes or motion_vectors is None or len(motion_vectors) == 0:
        return boxes

    mv = motion_vectors.to_ndarray()
    src_x, src_y = mv["src_x"], mv["src_y"]
    disp_x = mv["dst_x"].astype(np.int32) - src_x
    disp_y = mv["dst_y"].astype(np.int32) - src_y

    new_boxes = []
    for x, y, w, h in boxes:
        mask = (src_x >= x) & (src_x < x + w) & (src_y >= y) & (src_y < y + h)
        if np.any(mask):
            x = x + int(round(float(disp_x[mask].mean())))
            y = y + int(round(float(disp_y[mask].mean())))
            x = max(0, min(x, frame_width - 1))
            y = max(0, min(y, frame_height - 1))
            w = max(0, min(w, frame_width - x))
            h = max(0, min(h, frame_height - y))
        if w > 0 and h > 0:
            new_boxes.append((x, y, w, h))
    return new_boxes


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
            "Comma-separated ROI specifier(s), e.g. 'person' or 'person,dog,car'. "
            f"Each must be either a COCO class the {MODEL_ID} model knows (it's a "
            f"fixed-vocabulary COCO detector, not open-vocabulary), or the "
            f"reserved word '{TEXT_ROI_WORD}' to also detect on-screen text via "
            f"OpenCV's EAST detector, e.g. 'person,{TEXT_ROI_WORD}'. Every "
            "matching detection in a frame is boosted simultaneously (not just "
            "the single largest match), all sharing the same --qoffset. "
            "(default: %(default)s)"
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

    coco_words = [w for w in roi_words if w.lower() != TEXT_ROI_WORD]
    want_text = any(w.lower() == TEXT_ROI_WORD for w in roi_words)

    qoffset = Fraction(args.qoffset).limit_denominator(1000)

    # Load whatever detector(s) --roi actually needs, eagerly and with their
    # own status lines - on a first run the YOLO model download can take a
    # while and would otherwise look exactly like a hang once the frame loop
    # starts.
    if coco_words:
        print(f"Loading ROI detector model: {MODEL_ID} ...", flush=True)
        _get_model()
        print("Model ready.", flush=True)
    if want_text:
        print("Loading EAST text detector...", flush=True)
        _get_east_net()
        print("Text detector ready.", flush=True)

    in_container = av.open(args.video)
    in_stream = in_container.streams.video[0]
    # Ask the decoder to export the motion vectors it already computes during
    # motion compensation, so P/B-frames can reuse them for box propagation
    # instead of paying for a detector call on every single frame.
    in_stream.codec_context.flags2 |= Flags2.export_mvs
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

    # Timing buckets, excluding model loading (already done above, before any
    # of this runs). Decoding is timed around pulling each frame out of the
    # container's own decode generator, so it doesn't include the detect/
    # encode work done per iteration; detect and encode are timed around just
    # the calls that do that work.
    decode_time = 0.0
    detect_time = 0.0
    propagate_time = 0.0
    encode_time = 0.0
    i_frame_count = 0

    # Boxes carried over from the last I-frame, translated frame-by-frame via
    # motion vectors on the P/B-frames in between.
    tracked_boxes = []

    frame_iter = in_container.decode(in_stream)
    while True:
        t0 = time.perf_counter()
        try:
            frame = next(frame_iter)
        except StopIteration:
            break
        decode_time += time.perf_counter() - t0

        if frame.pict_type == PictureType.I:
            i_frame_count += 1
            t0 = time.perf_counter()
            # Detect on an RGB view of the frame - this doesn't touch
            # frame.ptr, so it's independent of whatever reformatting
            # happens below.
            rgb = frame.to_ndarray(format="rgb24")
            boxes = []
            if coco_words:
                boxes += get_boxes(rgb, roi_words=coco_words, confidence=args.conf)
            if want_text:
                boxes += get_text_boxes(rgb, confidence=args.conf)
            detect_time += time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            mvs = frame.side_data.get(SideDataType.MOTION_VECTORS)
            boxes = propagate_boxes(tracked_boxes, mvs, frame.width, frame.height)
            propagate_time += time.perf_counter() - t0
        tracked_boxes = boxes

        t0 = time.perf_counter()
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
        encode_time += time.perf_counter() - t0

        frame_idx += 1
        if total_frames > 0:
            pct = frame_idx / total_frames * 100
            print(f"\rFrame {frame_idx}/{total_frames} ({pct:5.1f}%)", end="", flush=True)
        else:
            print(f"\rFrame {frame_idx}", end="", flush=True)

    t0 = time.perf_counter()
    for packet in out_stream.encode():  # flush the encoder
        out_container.mux(packet)
    encode_time += time.perf_counter() - t0

    out_container.close()
    in_container.close()

    total_time = decode_time + detect_time + propagate_time + encode_time
    print(f"\nDone: {frame_idx} frame(s) processed -> {args.output}")
    print(
        f"Detected on {i_frame_count} I-frame(s), propagated boxes via "
        f"motion vectors on the other {frame_idx - i_frame_count} frame(s)."
    )
    print(
        f"Timing (model loading excluded): "
        f"decode {decode_time:.2f}s, detect {detect_time:.2f}s, "
        f"propagate {propagate_time:.2f}s, encode {encode_time:.2f}s, "
        f"total {total_time:.2f}s"
        + (f" ({frame_idx / total_time:.1f} fps)" if total_time > 0 else "")
    )

    # Average bitrate of the muxed output file - actual output duration
    # (frame_idx frames at the output stream's frame rate), not encode wall
    # time, so this reflects the video's real bitrate, not encoding speed.
    duration_sec = frame_idx / float(in_stream.average_rate)
    if duration_sec > 0:
        file_size_bits = os.path.getsize(args.output) * 8
        bitrate_kbps = file_size_bits / duration_sec / 1000
        print(f"Average bitrate: {bitrate_kbps:.1f} kbps ({duration_sec:.2f}s output)")


if __name__ == "__main__":
    main()
