#!/usr/bin/env bash
# Downloads the pretrained EAST text-detection model (frozen_east_text_detection.pb,
# ~92MB) used by get_text_boxes() in roi_encoder.py for --roi text.
#
# Not checked into git (too large for a repo this size) - fetched on demand
# here instead, same reasoning as native/build.sh fetching FFmpeg headers on
# demand rather than vendoring them.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/frozen_east_text_detection.pb"

if [ -f "$DEST" ]; then
  echo "Already present: $DEST"
  exit 0
fi

echo "Downloading EAST text detection model..."
curl -sL --fail -o "$DEST" \
  "https://raw.githubusercontent.com/oyyd/frozen_east_text_detection.pb/master/frozen_east_text_detection.pb"

echo "Saved: $DEST"
