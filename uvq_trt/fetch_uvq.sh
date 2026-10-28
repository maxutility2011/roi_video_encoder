#!/usr/bin/env bash
# Clones google/uvq (model code + checkpoints) next to this script, pinned to
# the revision the TensorRT export was written against.
set -euo pipefail

UVQ_REPO="https://github.com/google/uvq.git"
UVQ_REV="${UVQ_REV:-c204ecc04f093e90c941cb126c7ac3d5beca0414}"
DEST="$(cd "$(dirname "$0")" && pwd)/uvq"

if [ ! -d "$DEST/.git" ]; then
  git clone "$UVQ_REPO" "$DEST"
fi
git -C "$DEST" fetch --quiet origin "$UVQ_REV" || true
git -C "$DEST" checkout --quiet "$UVQ_REV"
echo "UVQ checked out at $DEST ($(git -C "$DEST" rev-parse --short HEAD))"
