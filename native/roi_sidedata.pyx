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


def attach_rois(VideoFrame frame, list regions, int qnum, int qden):
    """Attach one or more ROI regions to `frame`, all sharing the same QP
    offset qnum/qden (AVRational).

    `regions` is a list of (top, bottom, left, right) pixel-coordinate
    tuples, one per detected object - same convention as ffmpeg's addroi
    filter (bottom/right are exclusive-ish per libavutil's own docs, so pass
    whatever you'd have passed to addroi's y+h / x+w). Multiple regions are
    packed into a single side-data buffer as a real AVRegionOfInterest array
    (self_size is the per-entry stride libavcodec uses to walk it) - this is
    the same layout addroi itself would produce for overlapping/multiple
    regions, just built directly instead of through the filter graph.

    Call this at most once per frame, with ALL of that frame's regions in
    one call - side data of a given type is meant to appear once per frame;
    calling this twice on the same frame produces two separate
    REGIONS_OF_INTEREST blocks, which downstream consumers don't expect.

    IMPORTANT: call it BEFORE the frame's .side_data is ever accessed
    (including just to inspect/log it) - PyAV's _SideDataContainer builds
    and caches its view of the frame's side data on first access, so
    touching it first would hide what we attach here from any later
    Python-level read.
    """
    cdef lib.AVFrameSideData *sd
    cdef lib.AVRegionOfInterest *roi_array
    cdef size_t n = len(regions)
    cdef size_t i
    cdef int top, bottom, left, right

    if n == 0:
        return

    sd = lib.av_frame_new_side_data(
        frame.ptr,
        lib.AV_FRAME_DATA_REGIONS_OF_INTEREST,
        n * sizeof(lib.AVRegionOfInterest),
    )
    if sd == NULL:
        raise MemoryError("av_frame_new_side_data failed")

    roi_array = <lib.AVRegionOfInterest*> sd.data
    for i in range(n):
        top, bottom, left, right = regions[i]
        roi_array[i].self_size = sizeof(lib.AVRegionOfInterest)
        roi_array[i].top = top
        roi_array[i].bottom = bottom
        roi_array[i].left = left
        roi_array[i].right = right
        roi_array[i].qoffset.num = qnum
        roi_array[i].qoffset.den = qden
