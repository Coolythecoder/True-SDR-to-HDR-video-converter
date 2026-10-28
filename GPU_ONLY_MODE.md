# Optional GPU-only video processing (experimental)

This edition adds an **opt-in NVIDIA CUDA custom-export backend**. It does not
replace the tested CPU/OpenCL implementation. The checkbox is off by default.
The YouTube preset and the live preview keep their existing implementations.

## Start the Windows build

Extract the complete `HDRVideoStudio-GPU-Windows.zip` folder, including
`_internal`, and launch `HDRVideoStudio-GPU.exe`. Do not move the EXE away from
its bundled libraries. FFmpeg and FFprobe must still be available on PATH for
probing, audio, MP4 packaging and output checks. A compatible NVIDIA driver is
required; no driver is bundled. This first backend is NVIDIA-only, not AMD.

In **Export settings -> Custom export**, select **GPU-only video pipeline
(NVIDIA CUDA, experimental)**, then **Export custom**. It forces 10-bit NVENC
for this export and restores the previous custom encoder/bit-depth choices
when unticked. The quality slider selects native NVENC **constant QP**, with
preset P4 and no B-frames requested. These native encoder settings are explicit;
the result is not promised to be byte-identical to FFmpeg NVENC at the same
number. The YouTube button still uses the original x265 recipe, even when the
new checkbox is ticked.

Run a short representative clip first. Physical NVIDIA execution and speed
cannot be inferred from CPU-only CI or a successful Windows executable build.
The existing, verified hybrid mode remains available by unticking the option.
An export already running in another application instance is not changed.

## What stays on the GPU

Bulk video processing is:

`NVDEC -> native NV12/P010 CUDA surfaces -> float BT.709-to-linear-BT.2020 ->
existing creative grading -> full-pixel light reduction / PQ -> P010 -> NVENC`

Decoded and transformed movie frames stay in device memory. The light scan
uses every decoded pixel and frame, not a downsampled approximation. Only
scalar peak/average measurements return to the CPU; compressed NVENC bytes
also return for file writing. The second pass embeds the measured MaxCLL and
MaxFALL together with the existing original-SDR/virtual-creative mastering
policy as HEVC SEI. No stale manual metadata override is applied in this mode.

“GPU-only” describes the **bulk video pixel pipeline**, not zero CPU usage.
CPU tasks remain: the GUI and scheduling, packet/timestamp checks, tiny startup
reference tests, audio handling, file I/O, container muxing and output checks.
At startup a small generated test pattern is compared with the established CPU
colour reference and a native NVENC/P010 encode is checked for HDR10 signalling.
Final verification decodes one frame on the CPU to inspect SEI; it does not
reprocess the movie on the CPU. The live preview still uses the existing
SDR viewing transform and per-result CPU comparison safeguard.

## Input/output boundaries in v1

Supported inputs are explicitly tagged BT.709 SDR, H.264 or HEVC, progressive,
square-pixel 4:2:0 at 8 or 10 bits, with an explicit limited/full range and no
rotation. Even dimensions from 16x16 up to 7680x4320 are accepted by the input
contract; the actual decoder/encoder must also support them. Left/centre or
unspecified chroma locations are supported (unspecified uses left). Sources
must start at zero and have constant frame timing between 1 and 240 fps.
**144 fps is supported; it is not reduced to 60 fps.** Every input packet PTS
is checked with rational arithmetic before native encoding. VFR, gaps,
duplicate timestamps, unsupported profiles and unavailable capabilities stop
with a diagnostic instead of being silently retimed or processed on the CPU.

Output is HEVC Main 10, limited-range BT.2020/PQ, HDR10 mastering/content-light
SEI, in a fast-start `hvc1` MP4. The first audio track is copied when requested,
or encoded to AAC at 128 kb/s, following the custom audio choice. This is not a
multi-track archival remux: subtitles, chapters, other audio tracks and source
container metadata are not retained. Use the normal workflow when needed.

The native mode uses bilinear chroma reconstruction and centred 2x2 box-filtered
output chroma. Quantisation uses fixed 4x4 Bayer ordered dithering. This differs
from zimg's chroma filtering and error-diffusion dither; tiny pixel differences
are expected. Floating-point grading equations and reference-white scaling
are shared with the existing implementation. No new tone-mapping curve,
auto-exposure or saturation compensation is silently introduced.

## Failure handling and disk use

There is **no automatic CPU fallback** in this mode. A missing dependency,
failed CUDA reference test, unsupported source, driver/codec error or failed
output verification stops the export. Temporary files are removed and an
existing destination is not replaced. Successful output is published with an
atomic replace only after verification and a final cancellation check.

There are two GPU video passes (light scan, then encode). During final muxing,
the temporary HEVC elementary stream and MP4 coexist in the destination
folder, so allow space for approximately two encoded video copies in addition
to the source. The source must not be modified during export. The progress log
includes measured analysis/encoding fps; there is no advertised speedup until
measured on the actual machine.

Full-file timestamp scans, MP4 packaging/audio processing and output frame-count
verification have no fixed elapsed-time limit, so a long movie or slow disk
does not abort an otherwise valid export. These stages report elapsed time
every 15 seconds and remain cancellable. Short metadata probes and startup
checks retain their time limits; an error identifies the stage that failed.

## Run or rebuild from source

Use a separate Python 3.11 environment for the optional dependencies:

```powershell
python -m pip install -r requirements-gpu-only.txt
python gpu_only_studio.py
```

To create the Windows folder build:

```powershell
python -m pip install pyinstaller
python validate_cuda_build.py
python build_gpu_only.py
```

The build includes pinned CuPy, PyNvVideoCodec, CUDA 12.8 runtime/NVRTC libraries
and headers. Keep `licenses` and `BUILD_INFO.txt` with the distribution. The
original `SDR_to_HDR_cross_platform.py` entry point remains unchanged and does
not require these optional packages for its normal modes.

## Validation boundaries

Tests cover host execution of the same scalar colour/grade/PQ code against
actual FFmpeg, packet/timing contracts, SEI payloads, the P010 memory layout,
GUI mode isolation, cancellation/error reporting and real HDR10 video-copy
muxing at integer/fractional frame rates. Windows CI compiles generated kernels
with NVRTC for compute_120 and opens the packaged Qt application without a GPU.
Compilation is **not** execution. CI has no physical RTX 5080/NVDEC/NVENC device.
The application therefore performs its own startup reference/native-codec
checks on the selected machine before processing a full video.

To run the optional hardware regression tests on an NVIDIA machine with the
pinned dependencies and FFmpeg installed:

```powershell
$env:HDR_TEST_CUDA = '1'
python -m unittest discover -s tests -p test_gpu_only_native.py -v
```

These check actual P010 pixel preservation and short 8/10-bit exports, including
frame counts, 144 fps and fractional timing, colour tags and HDR10 metadata.
Source runs locate the installed CUDA runtime wheels before importing CuPy;
no machine-wide `CUDA_PATH` change is needed. The pinned PyNvVideoCodec version
uses a special P010 array type spelling and returns encoded packet records;
the GPU backend adapts those interfaces without copying video pixels to the CPU.

Primary API references:
- https://docs.nvidia.com/video-technologies/pynvvideocodec/pynvc-api-reference/encoder.html
- https://docs.nvidia.com/video-technologies/pynvvideocodec/pynvc-api-reference/decoder.html
- https://docs.cupy.dev/en/stable/install.html
