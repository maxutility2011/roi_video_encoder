"""Exports Google's UVQ 1.5 model to a single ONNX graph for TensorRT.

The graph input is `frames`, a (batch, 1080, 1920, 3) uint8 RGB tensor (what
FFmpeg's rgb24 output already is), and the output is `scores`, (batch, 1)
float32 per-frame UVQ 1.5 scores in [1, 5]. Only the batch dimension is
dynamic; UVQ itself always rescales frames to 1920x1080 before inference.

After exporting, the graph is run with ONNX Runtime and compared against
the original PyTorch UVQ1p5Core so a broken export is caught here, not
after a TensorRT build.

Example:
    python3 uvq_trt/export_onnx.py -o uvq_trt/uvq1p5.onnx
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import uvq1p5_trt  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-o", "--output", default=os.path.join(os.path.dirname(__file__), "uvq1p5.onnx"))
    parser.add_argument("--uvq_dir", default=uvq1p5_trt.DEFAULT_UVQ_DIR, help="Path to a google/uvq checkout")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset (>= 17 for a native LayerNormalization)")
    parser.add_argument("--skip_verify", action="store_true", help="Don't compare ONNX Runtime against PyTorch")
    parser.add_argument("--verify_batch", type=int, default=2, help="Frames used for the verification run")
    return parser.parse_args()


def verify(onnx_path, uvq_dir, batch, seed=0):
    """Compares ONNX Runtime on the exported graph with the unmodified UVQ."""
    import onnxruntime as ort

    rng = np.random.default_rng(seed)
    # Smooth random images + noise, so both the content and distortion paths
    # see something less degenerate than pure noise.
    base = rng.integers(0, 256, size=(batch, 9, 16, 3), dtype=np.uint8).repeat(120, 1).repeat(120, 2)
    noise = rng.integers(-20, 21, size=base.shape)
    frames = np.clip(base.astype(np.int32) + noise, 0, 255).astype(np.uint8)

    reference = uvq1p5_trt.load_uvq1p5(uvq_dir)
    with torch.inference_mode():
        video = torch.from_numpy(frames.astype(np.float32) * np.float32(2.0 / 255.0) - np.float32(1.0))
        expected = reference.uvq1p5_core(video.permute(0, 3, 1, 2).unsqueeze(1)).numpy()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    actual = session.run([uvq1p5_trt.OUTPUT_NAME], {uvq1p5_trt.INPUT_NAME: frames})[0]

    max_diff = float(np.max(np.abs(actual - expected)))
    print(f"PyTorch scores:      {expected.flatten().tolist()}")
    print(f"ONNX Runtime scores: {actual.flatten().tolist()}")
    print(f"max |diff| = {max_diff:.2e}")
    if max_diff > 1e-3:
        raise SystemExit("ONNX export does not match the PyTorch model")


def main():
    args = parse_args()
    model = uvq1p5_trt.build_fused_model(args.uvq_dir)
    example = torch.zeros(
        2, uvq1p5_trt.VIDEO_HEIGHT, uvq1p5_trt.VIDEO_WIDTH, uvq1p5_trt.VIDEO_CHANNELS, dtype=torch.uint8
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (example,),
            args.output,
            input_names=[uvq1p5_trt.INPUT_NAME],
            output_names=[uvq1p5_trt.OUTPUT_NAME],
            dynamic_axes={uvq1p5_trt.INPUT_NAME: {0: "batch"}, uvq1p5_trt.OUTPUT_NAME: {0: "batch"}},
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"Wrote {args.output} ({os.path.getsize(args.output) / 1e6:.1f} MB)")

    if not args.skip_verify:
        verify(args.output, args.uvq_dir, args.verify_batch)


if __name__ == "__main__":
    main()
