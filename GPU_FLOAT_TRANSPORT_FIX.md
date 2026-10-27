# GPU half-black image: float-plane transport repair

## The defect

The OpenCL grading kernels write three float32 grayscale planes. Before preview
or PQ encoding, FFmpeg merges them back into planar float RGB. Some legacy
FFmpeg `mergeplanes` implementations calculate each row's byte count as
`width * ((depth > 8) + 1)`. For a 32-bit float this copies **two bytes per
sample instead of four**: the left half of each image row is valid, while the
right half is uninitialised and can appear black. This is a CPU-side frame
copy after GPU grading, not evidence of a faulty NVIDIA GPU or colour grade.

Affected upstream implementations include the FFmpeg `n4.4` and `n5.1` tags:

- https://github.com/FFmpeg/FFmpeg/blob/n4.4/libavfilter/vf_mergeplanes.c
- https://github.com/FFmpeg/FFmpeg/blob/n5.1/libavfilter/vf_mergeplanes.c

Newer implementations use the actual sample byte count. A version string alone
is not a reliable check for patched vendor builds; the regression tests exercise
real pixels, including the right edge, rather than assuming a version is safe.

## Repair

`gpu_float_transport.merge_float_planes` duplicates each grayscale row with
`hstack`, merges the double-width planes, then explicitly crops the **left**
half back to the original width. The original row is entirely inside the bytes
copied by both the legacy two-byte and corrected four-byte implementations.

No pixel arithmetic, integer quantisation, scaling, brightness compensation or
colour reinterpretation occurs. The helper retains float32 headroom and restores
the already-known linear BT.2020 tags after the merge. This is a compatibility
repair of the shared GPU preview/export transport, not another CPU-grading
fallback. OpenCL still performs the grading. Temporary CPU buffers are wider,
so this does add copying overhead; it is not a measured speedup claim.

The content-light reduction outputs grayscale statistics directly and does not
need RGB reassembly, so that route is unchanged. YouTube's x265 encoder,
quality profiles, audio settings, HDR metadata policy, dimensions and timing
are unchanged. Per-result preview validation from PR #11 remains enabled.

## Validation

The new test covers all preview sizes, non-aligned geometry, two successive
frames and 3840x2160 float output. Exact byte equality checks negative values,
HDR headroom, channel order and the rightmost/bottommost pixels.

The legacy CI jobs run the original unprotected merge, require it to reproduce
the half-row loss, then require the repaired transport to preserve the entire
frame. The FFmpeg 4.4 job tests this CPU-side transport in isolation: that
runtime cannot download the grayf32le OpenCL frames needed by this application,
so it is not claimed to support the complete accelerated pipeline.

A separate Debian 12 / FFmpeg 5.1 job runs the full regression suite, including
OpenCL parity, full-size previews and HDR10 media tests, using PoCL software
OpenCL. The current-FFmpeg workflow continues to exercise the same shared
transport. Neither job relaxes the original pixel-comparison assertions.

PoCL executes the kernels on a CPU; passing it is not a physical NVIDIA
benchmark. The user's precise FFmpeg binary and RTX 5080 are not available in
CI, so confirmation on that machine still requires a rebuilt EXE and a preview
status of `OpenCL GPU grading`, not a CPU-fallback status.

## Update and check

Rebuild the complete repository, including the new Python module:

```powershell
git pull --ff-only origin main
python -m PyInstaller --clean --onefile --windowed SDR_to_HDR_cross_platform.py
```

Open the same SDR source and representative frame. Enable **GPU grading and
light analysis (Auto)**; toggling it off/on resets a previous preview rejection.
Check for a complete picture and `OpenCL GPU grading`. Before a long export,
compare a short accelerated output against the CPU version. The existing CPU
fallback remains available for other device or FFmpeg incompatibilities.
