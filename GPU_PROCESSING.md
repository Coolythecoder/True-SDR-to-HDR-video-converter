# GPU processing without changing YouTube export settings

## Controls

Open **Export settings → Processing acceleration**:

- **GPU grading and light analysis (Auto)** is on by default. It enables optional float32 OpenCL creative grading in the live preview and HDR exports, plus full-pixel content-light reductions on the GPU.
- **Try NVIDIA hardware decoding (NVDEC)** is on by default. It tests the actual input before using hardware decoding for HDR analysis/export. It does not change the output encoder. Preview input decoding remains software-based.
- **GPU encoding (NVENC)** remains a separate setting inside **Custom export**. It still does not apply to the YouTube preset.

Uncheck either processing option to disable that acceleration independently. Disabling both preserves the original CPU processing path. These choices are snapshotted when an export starts and locked with the other export controls.

## The YouTube contract is unchanged

The YouTube command builder and its verification rules are unchanged:

| Setting | Retained behaviour |
| --- | --- |
| Video encoder | CPU libx265, HEVC Main 10, 10-bit 4:2:0 |
| Balanced | CRF 20 / medium |
| High | CRF 16 / medium |
| Maximum | CRF 12 / slow |
| Colour | Actual SDR-to-linear-BT.2020 conversion, creative grade if selected, then PQ |
| HDR metadata | Mastering-target metadata and post-grade measured content-light values |
| Audio | Optional existing two-pass loudness normalization; stereo AAC, 48 kHz, 384 kb/s |
| Dimensions and timing | Existing source-preserving behaviour |
| Container | MP4 hvc1, fast-start, negative composition offsets, no edit lists |
| Validation | Existing post-encode YouTube HDR validation |

Processing acceleration changes neither the selected quality profile nor the chosen audio processing. It is not a substitution of NVENC for x265. Floating-point GPU calculations are checked against the CPU reference within numerical tolerances, not promised to be bit-identical. Rounding a measured peak to an integer nit can occasionally differ by one nit. Encoded file bytes can consequently differ without a changed quality setting.

## Which work moves to the GPU?

| Stage | Backend |
| --- | --- |
| Supported input decoding | Optional NVIDIA NVDEC, checked per source |
| Input colour management / conversion to float BT.2020 | CPU zscale, deliberately retained |
| Creative grading, highlight curve and chroma compression | OpenCL GPU when validated |
| Every-pixel luminance and frame-average reduction | OpenCL GPU when validated |
| Conversion from linear light to PQ / 10-bit YUV | CPU zscale, deliberately retained |
| YouTube HEVC encoding | CPU x265, unchanged |
| Audio / mux / output verification | CPU, unchanged |
| Preview grade | Optional OpenCL GPU |
| Preview SDR viewing transform and Qt display | Existing CPU path |

There are explicit CPU/GPU transfers. This is **not an end-to-end zero-copy CUDA pipeline**. The first version targets the expensive generic CPU grading equations and analysis readback, without replacing the established colour transforms with a different tone mapper or reducing analysis quality.

The content-light kernel processes every pixel of every decoded frame. It calculates maxima and sums for 16×16 tiles, including partial edge tiles. Only those compact statistics return through the analysis pipe; the CPU performs the final combination using the correct full-frame pixel count. This is not a downsampled or sampled estimate. At 3840×2160 the raw analysis pipe payload changes from 99,532,800 bytes per float RGB frame to 259,200 bytes of tile statistics (384× smaller); this is a data-volume calculation, **not a speedup claim**. The CPU-to-GPU upload of the source floats still exists.

## Requirements and selection

No extra Python package, CUDA Python binding, or downloaded model is required. The existing FFmpeg build must include OpenCL support and the `program_opencl`, `hwupload`, `hwdownload`, `extractplanes`, `mergeplanes`, `split` and `format` filters. Existing `zscale`, `geq`, x265 and preview filters remain required. The GPU driver must expose a usable OpenCL GPU device with the needed float image formats.

Automatic OpenCL selection tries an NVIDIA GPU first, then AMD, then another available GPU. It does not deliberately choose a CPU OpenCL device and mislabel that as GPU acceleration. Machines with multiple ambiguous devices may fall back to CPU; there is no manual device picker in this version.

NVDEC additionally needs a compatible NVIDIA driver and FFmpeg CUDA hardware-decoder support. It is only attempted for known 4:2:0 H.264, HEVC, VP9 or AV1 formats supported by the selection logic; actual hardware support varies. The source test compares a decoded frame with software decoding. Unsupported formats, missing drivers and probe differences use software decoding.

## Correctness and failure handling

The export worker executes a small GPU grading and reduction test against the CPU equations before selecting OpenCL. This happens away from the GUI thread. The same immutable grade is used for the content-light scan and encode. No reduction in bit depth or silent change of working colour space is used to obtain acceleration.

If accelerated analysis fails, the full scan is restarted on the CPU and that file's encode uses CPU processing. Partial measurements are discarded. Cancellation is not treated as a reason to restart work. A failed device check is logged, rather than silently claiming GPU use.

The live preview checks its initial accelerated image against the CPU rendering. Failed devices or excessive differences fall back to the CPU view. All work remains asynchronous; superseded results are discarded. Temporary generated shader files are retained until their process finishes, then cleaned up. Preview remains an **SDR tone-mapped approximation**, not a native HDR display surface.

A hardware failure during the final encoding pass is reported as a failed job; it is not silently converted into success or automatically restarted with potentially different metadata. Disable the relevant acceleration option and retry after inspecting the log.

## What to look for in the log

Successful use reports messages such as:

```
GPU colour processing: OpenCL ... (float grade and full-pixel analysis).
GPU input decoding: NVIDIA NVDEC; CPU zscale and selected encoder retained.
Processing backend: grade=OpenCL; decode=NVDEC; video encoder=libx265 (quality/audio unchanged).
```

With accurate conversion and no creative grade, `grade=CPU` is expected even when the GPU performs the content-light reduction. The preview status identifies `OpenCL GPU grading` when that path is active. Missing acceleration reports a CPU fallback. CPU x265 utilization during a YouTube export remains expected.

## Validation and performance boundary

Tests execute the real OpenCL kernels against the CPU reference, including all grading controls, values above SDR white, negative inputs, partial tiles and high peak ceilings. Additional tests check a real GPU-assisted HDR10 encode, decoded PQ luminance, measured SEI metadata, unchanged YouTube options, asynchronous preview, stale-result suppression, cancellation and CPU recovery.

The dedicated Linux CI job uses **PoCL software OpenCL on a CPU** to execute the kernels reproducibly. That validates the implementation and API integration; it is not a physical NVIDIA/AMD benchmark. Windows/macOS GUI/contract tests are distinct from hardware GPU validation. Native RTX 5080 execution and speed have not been measured in this environment.

Performance depends on source, resolution, enabled controls, GPU driver, transfer overhead and the retained x265 workload. Short previews can be limited by process/device startup. A CPU-bound x265 Slow encode will remain CPU-bound even after its surrounding processing is accelerated. Compare the same short source and settings on your machine before drawing conclusions about the speedup.

## Build

Use the existing entry point and rebuild from the complete repository. The generated OpenCL source is embedded in Python and written to a temporary directory at runtime, so no extra shader asset needs to be packaged:

```powershell
git pull --ff-only origin main
python -m PyInstaller --clean --onefile --windowed SDR_to_HDR_cross_platform.py
```

A previously built EXE is not updated automatically.

## Interface references

- FFmpeg OpenCL filters and `program_opencl`: https://ffmpeg.org/ffmpeg-filters.html
- NVIDIA FFmpeg hardware decoding/encoding: https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/ffmpeg-with-nvidia-gpu/index.html

The creative grading curve remains this application's implementation; use of OpenCL is not a claim of a new standardized HDR mastering transform.
