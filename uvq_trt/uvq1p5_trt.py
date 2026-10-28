"""Shared pieces for running Google's UVQ 1.5 model with TensorRT.

UVQ 1.5 (https://github.com/google/uvq) is three PyTorch networks:
ContentNet (EfficientNet-B0 on the frame resized to 256x256), DistortionNet
(EfficientNet-B0 on nine 360x640 patches of the 1080p frame) and
AggregationNet (fuses both feature maps into one score in [1, 5]).

UVQ1p5Fused below wires those three into a single static-shape graph that
takes raw uint8 RGB frames, so the whole model - including the [-1, 1]
normalization, the patch split and the 256x256 resize - becomes one ONNX
graph and therefore one TensorRT engine.
"""

import os
import subprocess
import sys
import tempfile
import types

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

DEFAULT_UVQ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uvq")

VIDEO_HEIGHT = 1080
VIDEO_WIDTH = 1920
VIDEO_CHANNELS = 3
PATCH_HEIGHT = 360
PATCH_WIDTH = 640
NUM_PATCHES_Y = VIDEO_HEIGHT // PATCH_HEIGHT
NUM_PATCHES_X = VIDEO_WIDTH // PATCH_WIDTH
FEATURE_CHANNELS = 128
CONTENT_FEATURE_SIZE = 8  # ContentNet: 256x256 input / 32
PATCH_FEATURE_SIZE = 8  # DistortionNet: (12x20 after /32) max-pooled by (5, 13)
AGGREGATION_SIZE = 4  # AggregationNet resizes both feature maps to 4x4

INPUT_NAME = "frames"
OUTPUT_NAME = "scores"


def add_uvq_to_path(uvq_dir=DEFAULT_UVQ_DIR):
    """Makes a google/uvq checkout importable (its model code + utils/)."""
    uvq_dir = os.path.abspath(uvq_dir)
    if not os.path.isfile(os.path.join(uvq_dir, "uvq1p5_pytorch", "utils", "uvq1p5.py")):
        raise SystemExit(
            f"No UVQ checkout found at {uvq_dir}. Run uvq_trt/fetch_uvq.sh "
            "or pass --uvq_dir pointing at a clone of https://github.com/google/uvq"
        )
    for path in (uvq_dir, os.path.join(uvq_dir, "utils")):
        if path not in sys.path:
            sys.path.insert(0, path)
    return uvq_dir


def load_uvq1p5(uvq_dir=DEFAULT_UVQ_DIR):
    """Returns the original pretrained UVQ1p5 PyTorch model, in eval mode."""
    add_uvq_to_path(uvq_dir)
    from uvq1p5_pytorch.utils import uvq1p5

    model = uvq1p5.UVQ1p5()
    model.eval()
    return model


class UVQ1p5Fused(nn.Module):
    """UVQ 1.5 as one graph: (B, 1080, 1920, 3) uint8 RGB -> (B, 1) score.

    Numerically this is UVQ1p5Core applied to frames normalized exactly like
    utils/video_reader.py does (x * 2/255 - 1), but written with static
    shapes and a symbolic batch dimension so it exports cleanly to ONNX.
    """

    def __init__(self, uvq1p5_model):
        super().__init__()
        self.content = uvq1p5_model.content_net.model
        self.distortion = uvq1p5_model.distortion_net.model
        self.aggregation = uvq1p5_model.aggregation_net.model
        # AggregationNet's AdaptiveAvgPool2d((4, 4)) "resizer" is shared by the
        # 8x8 content and 24x24 distortion maps. The ONNX exporter can't size
        # it (spatial dims get lost behind the asymmetric same-padding), so
        # the equivalent fixed average pools are applied in forward() instead.
        self.aggregation.resizer = nn.Identity()

    def forward(self, frames):
        x = frames.permute(0, 3, 1, 2).to(torch.float32)
        x = x * (2.0 / 255.0) - 1.0

        # ContentNetCore: bilinear resize to 256x256 + EfficientNet -> (B, 128, 8, 8).
        content = self.content(x)

        # DistortionNet: split into a 3x3 grid of 360x640 patches, run them as
        # one batch, then stitch the (8, 8) patch features into (B, 128, 24, 24).
        patches = x.reshape(-1, VIDEO_CHANNELS, NUM_PATCHES_Y, PATCH_HEIGHT, NUM_PATCHES_X, PATCH_WIDTH)
        patches = patches.permute(0, 2, 4, 1, 3, 5)
        patches = patches.reshape(-1, VIDEO_CHANNELS, PATCH_HEIGHT, PATCH_WIDTH)
        patch_features = self.distortion.avgpool[0](self.distortion.features(patches))
        c, fh, fw = FEATURE_CHANNELS, PATCH_FEATURE_SIZE, PATCH_FEATURE_SIZE
        distortion = patch_features.reshape(-1, NUM_PATCHES_Y, NUM_PATCHES_X, c, fh, fw)
        distortion = distortion.permute(0, 3, 1, 4, 2, 5)
        distortion = distortion.reshape(-1, c, NUM_PATCHES_Y * fh, NUM_PATCHES_X * fw)

        content = F.avg_pool2d(content, CONTENT_FEATURE_SIZE // AGGREGATION_SIZE)
        distortion = F.avg_pool2d(distortion, NUM_PATCHES_Y * PATCH_FEATURE_SIZE // AGGREGATION_SIZE)
        return self.aggregation({"content": content, "distortion": distortion})


def _static_same_padding_forward(self, x):
    if self.static_pad is not None:
        x = F.pad(x, self.static_pad)
    return self.conv(x)


def freeze_same_padding(model, example):
    """Replaces UVQ's runtime "same"-padding computation with constants.

    Conv2dSamePadding derives its padding from x.size() on every call, which
    exports as a chain of Shape/Gather/Max ops in front of each convolution.
    All spatial sizes are fixed here, so the padding is recorded once with a
    dry run and baked in: symmetric padding moves into the Conv itself (lets
    TensorRT fuse it), asymmetric padding stays a single constant Pad.
    """
    from uvq1p5_pytorch.utils import custom_nn_layers

    input_hw = {}
    hooks = [
        m.register_forward_pre_hook(lambda mod, args: input_hw.__setitem__(mod, tuple(args[0].shape[2:])))
        for m in model.modules()
        if isinstance(m, custom_nn_layers.Conv2dSamePadding)
    ]
    with torch.inference_mode():
        model(example)
    for hook in hooks:
        hook.remove()

    for m, (ih, iw) in input_hw.items():
        (kh, kw), (sh, sw) = m.kernel_size, m.stride
        ph = max((ih - 1) // sh * sh + kh - ih, 0)
        pw = max((iw - 1) // sw * sw + kw - iw, 0)
        top, left = ph // 2, pw // 2
        bottom, right = ph - top, pw - left
        if top == bottom and left == right:
            m.conv.padding = (top, left)
            m.static_pad = None
        else:
            m.static_pad = (left, right, top, bottom)
        m.forward = types.MethodType(_static_same_padding_forward, m)
    return len(input_hw)


def build_fused_model(uvq_dir=DEFAULT_UVQ_DIR):
    """Pretrained UVQ1p5Fused with constant padding, ready for export."""
    fused = UVQ1p5Fused(load_uvq1p5(uvq_dir)).eval()
    example = torch.zeros(1, VIDEO_HEIGHT, VIDEO_WIDTH, VIDEO_CHANNELS, dtype=torch.uint8)
    freeze_same_padding(fused, example)
    return fused


# ---------------------------------------------------------------------------
# Video decoding: identical FFmpeg pipeline to UVQ's iter_video_batches_1p5,
# but frames stay uint8 (normalization happens inside the engine), which is
# 4x less host memory traffic and host->device copy than float32.
# ---------------------------------------------------------------------------
def iter_uint8_batches(
    filepath,
    video_length,
    transpose=False,
    video_fps=1,
    batch_size=24,
    ffmpeg_path="ffmpeg",
    hwaccel=None,
):
    """Yields (frames, num_real_frames) with frames (N, 1080, 1920, 3) uint8.

    Same frame selection and padding rules as UVQ: exactly
    video_length * video_fps frames, truncated decodes padded with black
    (0, which normalizes to UVQ's -1.0 padding value).
    """
    import video_reader  # from the UVQ checkout (utils/video_reader.py)

    frame_size = VIDEO_WIDTH * VIDEO_HEIGHT * VIDEO_CHANNELS
    expected_frames = video_length * video_fps
    command = video_reader._video_1p5_command(
        filepath, transpose, video_fps, VIDEO_HEIGHT, VIDEO_WIDTH, ffmpeg_path
    )
    if hwaccel:
        # Decoding is bit-exact across decoders, so this only offloads work.
        command[1:1] = ["-hwaccel", hwaccel]

    num_real_frames = 0
    emitted_frames = 0
    with tempfile.TemporaryFile() as error_file:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=error_file)
        completed_read = False
        try:
            while True:
                rgb = process.stdout.read(batch_size * frame_size)
                if not rgb:
                    break
                if len(rgb) % frame_size != 0:
                    raise RuntimeError(
                        f"FFmpeg returned an incomplete RGB frame: {len(rgb)} bytes "
                        f"is not divisible by {frame_size}"
                    )
                decoded = len(rgb) // frame_size
                num_real_frames += decoded
                to_emit = min(decoded, expected_frames - emitted_frames)
                if to_emit <= 0:
                    continue
                # bytearray: torch.from_numpy() needs a writable buffer.
                frames = np.frombuffer(bytearray(rgb), dtype=np.uint8, count=to_emit * frame_size)
                emitted_frames += to_emit
                yield frames.reshape(to_emit, VIDEO_HEIGHT, VIDEO_WIDTH, VIDEO_CHANNELS), to_emit
            completed_read = True
            return_code = process.wait()
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                process.wait()

        if completed_read and return_code != 0:
            error_file.seek(0)
            raise subprocess.CalledProcessError(
                return_code, command, stderr=error_file.read().decode(errors="replace")
            )

    if num_real_frames < 1:
        raise RuntimeError(f"Decoding failed to output a single frame for '{filepath}'")

    while emitted_frames < expected_frames:
        n = min(batch_size, expected_frames - emitted_frames)
        emitted_frames += n
        yield np.zeros((n, VIDEO_HEIGHT, VIDEO_WIDTH, VIDEO_CHANNELS), dtype=np.uint8), 0


# ---------------------------------------------------------------------------
# TensorRT runtime (TensorRT >= 8.6 tensor-address API, incl. TensorRT 10).
# ---------------------------------------------------------------------------
class TRTEngine:
    """Runs a serialized UVQ 1.5 engine on CUDA torch tensors."""

    def __init__(self, engine_path, device="cuda"):
        import tensorrt as trt

        self.device = torch.device(device)
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(
                f"Failed to deserialize {engine_path}. Engines are tied to the GPU "
                "model and TensorRT version they were built with - rebuild it on "
                "this machine with uvq_trt/build_engine.py."
            )
        self.context = self.engine.create_execution_context()

        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_name = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.output_name = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        self.input_dtype = torch.from_numpy(
            np.empty(0, dtype=trt.nptype(self.engine.get_tensor_dtype(self.input_name)))
        ).dtype
        min_shape, _, max_shape = self.engine.get_tensor_profile_shape(self.input_name, 0)
        self.min_batch, self.max_batch = int(min_shape[0]), int(max_shape[0])

    def __call__(self, frames):
        """frames: (N, 1080, 1920, 3) uint8 tensor (any device) -> (N, 1) float32 on GPU."""
        frames = frames.to(self.device, dtype=self.input_dtype, non_blocking=True).contiguous()
        outputs = []
        for start in range(0, frames.shape[0], self.max_batch):
            chunk = frames[start:start + self.max_batch]
            n = chunk.shape[0]
            if n < self.min_batch:  # pad up to the profile minimum, drop afterwards
                pad = chunk.new_zeros((self.min_batch - n,) + tuple(chunk.shape[1:]))
                chunk = torch.cat([chunk, pad])
            self.context.set_input_shape(self.input_name, tuple(chunk.shape))
            out = torch.empty(
                tuple(self.context.get_tensor_shape(self.output_name)),
                dtype=torch.float32,
                device=self.device,
            )
            self.context.set_tensor_address(self.input_name, chunk.data_ptr())
            self.context.set_tensor_address(self.output_name, out.data_ptr())
            stream = torch.cuda.current_stream(self.device)
            if not self.context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("TensorRT execution failed")
            outputs.append(out[:n])
        return torch.cat(outputs)
