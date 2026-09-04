#!/usr/bin/env bash
# Builds native/roi_sidedata.so: a small Cython extension that attaches real
# AV_FRAME_DATA_REGIONS_OF_INTEREST side data to a PyAV VideoFrame, so
# libx264 gets a genuine per-frame ROI QP bias without going through
# ffmpeg's addroi filter (which can't vary its region per frame at all).
#
# Why this can't just be `pip install` or a normal setup.py build:
#  - PyAV bundles its OWN private copy of the FFmpeg shared libraries, and
#    the pxd declarations needed to write a Cython extension against PyAV's
#    Frame objects (av/*.pxd -> `cimport libav as lib`) are NOT included in
#    PyAV's installed wheel - only in its sdist.
#  - Compiling those pxd declarations requires real FFmpeg C headers on
#    disk, and they must be the exact FFmpeg release PyAV bundled - not
#    whatever version `apt` happens to offer - or you risk a genuine
#    struct-layout/ABI mismatch against PyAV's actual runtime library.
#
# So this script, every time it runs:
#  1. Asks the venv's installed `av` package which FFmpeg libs it bundles
#     (avutil/avcodec/avformat version triple).
#  2. Maps that triple to the matching FFmpeg git tag (see FFMPEG_TAG_MAP
#     below - add an entry here if you ever upgrade the `av` package and
#     this script complains it doesn't recognize the new version triple;
#     find the tag by diffing libavutil/version.h across candidate tags,
#     e.g. `curl -s https://raw.githubusercontent.com/FFmpeg/FFmpeg/n7.2/libavutil/version.h`).
#  3. Downloads that FFmpeg tag's source and runs just enough of `./configure`
#     to generate its auto-generated headers (avconfig.h, config.h) - no
#     actual FFmpeg build happens, this is headers-only and takes seconds.
#  4. Downloads the matching `av` sdist (source distribution) for its
#     vendored include/libav.pxd tree, and patches in the two declarations
#     it's missing (AVRegionOfInterest, av_frame_new_side_data) - both
#     copied verbatim from the real header fetched in step 3.
#  5. Cythonizes and compiles roi_sidedata.pyx against all of the above,
#     linking against PyAV's own bundled libavutil .so (not system ffmpeg).
#
# Everything downloaded/generated lives under native/.build/ (gitignored,
# safe to `rm -rf` and rerun). No `sudo`, no system package installation.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$HERE/.build"
PYTHON="${PYTHON:-python3}"

mkdir -p "$BUILD"

echo "== Checking installed 'av' package =="
read -r AV_VERSION AVUTIL_VER AVCODEC_VER AVFORMAT_VER <<EOF
$("$PYTHON" - <<'PYEOF'
import av
u = av.library_versions["libavutil"]
c = av.library_versions["libavcodec"]
f = av.library_versions["libavformat"]
print(av.__version__, ".".join(map(str, u)), ".".join(map(str, c)), ".".join(map(str, f)))
PYEOF
)
EOF
echo "av $AV_VERSION bundles libavutil $AVUTIL_VER / libavcodec $AVCODEC_VER / libavformat $AVFORMAT_VER"

# Map (avutil, avcodec, avformat) version triples -> matching FFmpeg git tag.
# Verified by diffing libavutil/version.h (etc.) across candidate tags - see
# the header comment above for how to add a new row.
case "$AVUTIL_VER.$AVCODEC_VER.$AVFORMAT_VER" in
  "59.39.100.61.19.100.61.7.100")
    FFMPEG_TAG="n7.1"
    ;;
  *)
    echo "ERROR: no known FFmpeg tag mapping for avutil $AVUTIL_VER / avcodec $AVCODEC_VER / avformat $AVFORMAT_VER." >&2
    echo "Add a mapping in build.sh - find the matching tag by comparing" >&2
    echo "libavutil/version.h across FFmpeg release tags on GitHub." >&2
    exit 1
    ;;
esac
echo "-> matches FFmpeg tag $FFMPEG_TAG"

# --- Step 3: real FFmpeg headers, configured (not built) ---
FFDIR="$BUILD/ffmpeg-$FFMPEG_TAG"
if [ ! -f "$FFDIR/libavutil/avconfig.h" ]; then
  echo "== Fetching FFmpeg $FFMPEG_TAG source (headers only) =="
  mkdir -p "$FFDIR"
  curl -sL --fail "https://github.com/FFmpeg/FFmpeg/archive/refs/tags/${FFMPEG_TAG}.tar.gz" \
    | tar xz -C "$BUILD"
  mv "$BUILD/FFmpeg-${FFMPEG_TAG#n}"/* "$FFDIR"/ 2>/dev/null \
    || mv "$BUILD"/FFmpeg-*/* "$FFDIR"/
  rmdir "$BUILD"/FFmpeg-* 2>/dev/null || true

  echo "== Running FFmpeg's ./configure (headers/config only, no build) =="
  (cd "$FFDIR" && ./configure \
    --disable-everything --disable-doc --disable-programs \
    --disable-avdevice --disable-avfilter --disable-swresample \
    --disable-network --disable-x86asm --disable-inline-asm \
    >/dev/null)
else
  echo "== FFmpeg $FFMPEG_TAG headers already present, skipping fetch =="
fi

# --- Step 4: PyAV's vendored pxd tree, patched ---
VENDOR="$BUILD/vendor_include"
if [ ! -f "$VENDOR/libav.pxd" ]; then
  echo "== Fetching av==$AV_VERSION sdist for its vendored libav.pxd tree =="
  rm -rf "$BUILD/av-src"
  mkdir -p "$BUILD/av-src"
  curl -sL --fail "https://files.pythonhosted.org/packages/source/a/av/av-${AV_VERSION}.tar.gz" \
    | tar xz -C "$BUILD/av-src"
  rm -rf "$VENDOR"
  cp -r "$BUILD/av-src/av-${AV_VERSION}/include" "$VENDOR"
fi

FRAME_PXD="$VENDOR/libavutil/frame.pxd"
if ! grep -q "av_frame_new_side_data" "$FRAME_PXD"; then
  echo "== Patching in AVRegionOfInterest / av_frame_new_side_data =="
  cat >> "$FRAME_PXD" <<'PXD'

    # Not in PyAV's vendored declarations - added by hand against the real
    # libavutil/frame.h for the matching FFmpeg tag (see build.sh).
    cdef struct AVRegionOfInterest:
        unsigned int self_size
        int top
        int bottom
        int left
        int right
        AVRational qoffset

    cdef AVFrameSideData* av_frame_new_side_data(AVFrame *frame, AVFrameSideDataType type, size_t size)
PXD
fi

# --- Step 5: cythonize + compile, linking PyAV's bundled libavutil ---
echo "== Locating PyAV's bundled libavutil =="
read -r SITE_PKGS AVLIBS AVUTIL_SO <<EOF
$("$PYTHON" - <<'PYEOF'
import av, os, glob
site_pkgs = os.path.dirname(os.path.dirname(av.__file__))
avlibs = os.path.join(site_pkgs, "av.libs")
matches = glob.glob(os.path.join(avlibs, "libavutil*.so*"))
if not matches:
    raise SystemExit("no libavutil*.so* found under " + avlibs)
print(site_pkgs, avlibs, os.path.basename(matches[0]))
PYEOF
)
EOF
echo "site-packages: $SITE_PKGS"
echo "av.libs:       $AVLIBS"
echo "libavutil:     $AVUTIL_SO"

PYINC="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_path("include"))')"

echo "== Cythonizing =="
"$PYTHON" -m cython -3 -I "$VENDOR" -I "$SITE_PKGS" \
  "$HERE/roi_sidedata.pyx" -o "$BUILD/roi_sidedata.c"

echo "== Compiling =="
# --disable-new-dtags: emit the old-style DT_RPATH instead of the modern
# DT_RUNPATH. DT_RUNPATH only helps resolve THIS .so's own direct deps
# (libavutil); it does NOT propagate to libavutil's own transitive deps
# (e.g. bundled libdrm), so without this, import fails at that second hop.
# PyAV's own bundled .so files use the same old-style RPATH for exactly
# this reason - confirm with `readelf -d <av package dir>/utils*.so`.
gcc -shared -fPIC -O2 \
  -I "$PYINC" -I "$FFDIR" \
  "$BUILD/roi_sidedata.c" -o "$HERE/roi_sidedata.so" \
  -L "$AVLIBS" -l:"$AVUTIL_SO" \
  -Wl,-rpath,"$AVLIBS" -Wl,--disable-new-dtags

echo "== Smoke test =="
(cd "$HERE" && "$PYTHON" -c "
import roi_sidedata, av
f = av.VideoFrame(width=16, height=16, format='yuv420p')
roi_sidedata.attach_roi(f, top=0, bottom=16, left=0, right=16, qnum=-1, qden=2)
sd = f.side_data[0]
assert len(bytes(sd)) == 28, 'unexpected side-data size'
print('roi_sidedata.so OK:', f.side_data[0])
")

echo
echo "Built: $HERE/roi_sidedata.so"
