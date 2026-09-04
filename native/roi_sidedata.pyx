"""Attach a real AV_FRAME_DATA_REGIONS_OF_INTEREST side-data blob directly to
a PyAV VideoFrame's underlying AVFrame, so libx264 (via PyAV's bundled
libavcodec) applies a genuine per-frame ROI QP bias on encode.

This exists because FFmpeg's `addroi` filter cannot vary its region per frame
through any CLI mechanism in this build (its x/y/w/h options are evaluated
once at filter-graph init, not per frame, and aren't runtime-tunable via
sendcmd) - confirmed empirically. Attaching the side data directly to the
frame object bypasses the filter graph entirely and can never desync from the
frame it was detected on.

Built against PyAV's own vendored libav.pxd (from its sdist, not its wheel -
the installed wheel doesn't ship libav.pxd at all) plus two declarations that
sdist is missing (AVRegionOfInterest, av_frame_new_side_data), compiled
against real FFmpeg headers from the exact release PyAV bundles. See
build.sh for how all of that gets assembled - do not hand-edit the vendored
tree under .build/, it's regenerated from scratch on every build.
"""
cimport libav as lib
from av.video.frame cimport VideoFrame


def attach_roi(VideoFrame frame, int top, int bottom, int left, int right,
               int qnum, int qden):
    """Attach one ROI region to `frame` with QP offset qnum/qden (AVRational).

    top/bottom/left/right are pixel coordinates, same convention as ffmpeg's
    addroi filter (bottom/right are exclusive-ish per libavutil's own docs -
    match whatever region you'd have passed to addroi's y+h / x+w).

    IMPORTANT: call this exactly once per frame, and call it BEFORE the
    frame's .side_data is ever accessed (including just to inspect/log it) -
    PyAV's _SideDataContainer builds and caches its view of the frame's side
    data on first access, so touching it first would hide what we attach
    here from any later Python-level read.
    """
    cdef lib.AVFrameSideData *sd
    cdef lib.AVRegionOfInterest *roi

    sd = lib.av_frame_new_side_data(
        frame.ptr,
        lib.AV_FRAME_DATA_REGIONS_OF_INTEREST,
        sizeof(lib.AVRegionOfInterest),
    )
    if sd == NULL:
        raise MemoryError("av_frame_new_side_data failed")

    roi = <lib.AVRegionOfInterest*> sd.data
    roi.self_size = sizeof(lib.AVRegionOfInterest)
    roi.top = top
    roi.bottom = bottom
    roi.left = left
    roi.right = right
    roi.qoffset.num = qnum
    roi.qoffset.den = qden
