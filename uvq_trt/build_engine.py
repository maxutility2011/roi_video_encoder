"""Builds (serializes) a TensorRT engine from the UVQ 1.5 ONNX graph.

A TensorRT engine is specific to the GPU model and TensorRT version it was
built with, so run this on the machine that will run inference.

Example:
    python3 uvq_trt/build_engine.py uvq_trt/uvq1p5.onnx -o uvq_trt/uvq1p5_fp16.engine --fp16

Equivalent trtexec command:
    trtexec --onnx=uvq_trt/uvq1p5.onnx --saveEngine=uvq_trt/uvq1p5_fp16.engine --fp16 \
        --minShapes=frames:1x1080x1920x3 --optShapes=frames:24x1080x1920x3 \
        --maxShapes=frames:24x1080x1920x3
"""

import argparse
import os
import sys
import time

import tensorrt as trt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uvq1p5_trt import INPUT_NAME, VIDEO_CHANNELS, VIDEO_HEIGHT, VIDEO_WIDTH  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("onnx", help="ONNX file written by export_onnx.py")
    parser.add_argument("-o", "--output", default=None, help="Engine path (default: next to the ONNX file)")
    parser.add_argument("--fp16", action="store_true", help="Allow FP16 kernels (much faster on Tensor Core GPUs)")
    parser.add_argument("--min_batch", type=int, default=1)
    parser.add_argument("--opt_batch", type=int, default=24, help="Batch size TensorRT tunes kernels for")
    parser.add_argument("--max_batch", type=int, default=24, help="Largest batch the engine accepts")
    parser.add_argument("--workspace_gb", type=float, default=4.0, help="Builder scratch memory limit")
    parser.add_argument("--timing_cache", default=None, help="Optional timing cache file to speed up rebuilds")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.min_batch <= args.opt_batch <= args.max_batch:
        parser.error("need 1 <= --min_batch <= --opt_batch <= --max_batch")
    if args.output is None:
        args.output = os.path.splitext(args.onnx)[0] + ("_fp16" if args.fp16 else "_fp32") + ".engine"
    return args


def main():
    args = parse_args()
    logger = trt.Logger(trt.Logger.VERBOSE if args.verbose else trt.Logger.INFO)
    builder = trt.Builder(logger)

    # TensorRT 10 is always explicit-batch; 8.x needs the flag.
    flags = 0
    if int(trt.__version__.split(".")[0]) < 10:
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)

    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(args.onnx):
        for i in range(parser.num_errors):
            print(parser.get_error(i), file=sys.stderr)
        raise SystemExit(f"Failed to parse {args.onnx}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gb * (1 << 30)))
    if args.fp16:
        if not builder.platform_has_fast_fp16:
            print("Warning: this GPU has no fast FP16; the engine may not be faster", file=sys.stderr)
        config.set_flag(trt.BuilderFlag.FP16)

    cache = None
    if args.timing_cache:
        blob = open(args.timing_cache, "rb").read() if os.path.exists(args.timing_cache) else b""
        cache = config.create_timing_cache(blob)
        config.set_timing_cache(cache, ignore_mismatch=False)

    frame_shape = (VIDEO_HEIGHT, VIDEO_WIDTH, VIDEO_CHANNELS)
    profile = builder.create_optimization_profile()
    profile.set_shape(
        INPUT_NAME,
        (args.min_batch,) + frame_shape,
        (args.opt_batch,) + frame_shape,
        (args.max_batch,) + frame_shape,
    )
    config.add_optimization_profile(profile)

    print(f"Building {'FP16' if args.fp16 else 'FP32'} engine, batch {args.min_batch}..{args.max_batch} "
          f"(opt {args.opt_batch}) with TensorRT {trt.__version__} - this takes a few minutes")
    start = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("TensorRT engine build failed (see log above)")

    with open(args.output, "wb") as f:
        f.write(serialized)
    if cache is not None:
        with open(args.timing_cache, "wb") as f:
            f.write(cache.serialize())
    print(f"Wrote {args.output} ({os.path.getsize(args.output) / 1e6:.1f} MB) in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
