# SDR to HDR Video Converter (Cross-Platform)

A Python + PyQt5 desktop application to map SDR video into 10-bit BT.2020/PQ output using FFmpeg's zscale filter. It supports NVENC or CPU encoding, optional x265 HDR10 static metadata, batch mode, **optional audio passthrough**, and real-time logging. This is a display-referred SDR-to-HDR signal conversion; it cannot recreate highlights already clipped in the original SDR recording.

---

## 🚀 Features

- ✅ **SDR to HDR video conversion** using FFmpeg
- ⚡ **GPU acceleration** with NVENC (with auto fallback to CPU/x265)
- 🧠 **Auto encoder switching** based on FFmpeg availability
- 🎛️ **Correct HDR colour transform**: source SDR transfer → linear BT.2020 RGB → 10-bit PQ/BT.2020 YUV
- 🔢 **Adjustable HDR reference white**: 80–400 nits (200-nit default, based on MovieLabs guidance)
- 🎚️ **Constant Quality (CRF) slider**
- 🌈 **Accurate BT.709 → BT.2020 primaries conversion**, with metadata probing for supported SDR input profiles
- 🔊 **Audio passthrough toggle**: retain original audio or re-encode to AAC
- 📊 **Static HDR10 metadata with libx265**: BT.709 mastering primaries, appropriate white luminance, MaxCLL and unknown MaxFALL; NVENC carries PQ/BT.2020 colour signalling but does not embed the x265-specific mastering SEI
- 📝 **`-x265-params`** integration for HDR metadata (compatible with libx265)
- 🖥️ **Real-time FFmpeg log viewer**
- 📦 **Batch conversion mode** (process MP4/MOV/MKV files in a folder, one by one)
- ❌ **Safe cancellation** of conversions (Windows & Linux)

---

## 🖥️ GUI Overview

- **Input/Output**: Select a video (or folder in batch mode) and an existing output folder. Output filenames are generated as `<input>_HDR.mp4`.
- **HDR mapping**: 10-bit mode performs the actual pixel transform to PQ; adjust the 200-nit default reference white cautiously
- **Encoding Options**:
  - Bit depth: 10-bit (HDR PQ, default) or 8-bit (SDR-only)
  - CRF slider (0–51)
  - GPU toggle (NVENC)
  - **Audio Passthrough**: Optional toggle to copy original audio stream
- **HDR Metadata Options**:
  - For CPU/libx265 10-bit output, embed default static metadata or override MaxCLL/MaxFALL when known
  - NVENC output includes BT.2020/PQ tags but not x265 static HDR10 mastering SEI
- **Log Output**: Real-time FFmpeg log monitor
- **Execution**: Batch mode, preview mode, and a cancel-safe exit

---

## 🛠️ Requirements

- **Python 3.7+**
- **FFmpeg** (in PATH, with the **zscale/libzimg filter** and libx265 or hevc_nvenc for HDR export)
- **FFprobe** (recommended; shipped with most FFmpeg distributions to detect source colour tags)
- Python packages:
  - `PyQt5`
  - `numpy`
  - `opencv-python`

### Install Dependencies

```bash
pip install PyQt5 numpy opencv-python
```

---

## ▶️ How to Run

```bash
python SDR_to_HDR_cross_platform.py
```

---

## 🏗️ Build Executable (Optional)

To build a standalone executable:

```bash
pip install pyinstaller
pyinstaller --onefile --windowed SDR_to_HDR_cross_platform.py
```

> Output will be in the `dist/` directory.

---

## 📂 Supported Input Formats

- `.mp4`
- `.mov`
- `.mkv`

---

## ⚠️ Notes

- **Never relabel unconverted SDR as PQ HDR.** In 10-bit mode, the application automatically applies the zscale BT.709/other-supported-SDR to display-linear BT.2020 to PQ conversion before encoding.
- 10-bit HDR outputs use `_HDR.mp4`. 8-bit SDR outputs use `_SDR.mp4` and are not tagged as HDR.
- HDR PQ reference white defaults to 200 nits: the original SDR image is mapped into an HDR container without inventing detail above the SDR source's available range.
- CPU/libx265 can embed static ST 2086 + MaxCLL/MaxFALL SEI; NVENC output is tagged PQ/BT.2020 but does **not** embed those x265-specific SEI fields.
- Only tested SDR colour formats can be converted. Declared HDR PQ/HLG inputs and unsupported declared primaries/transfers/matrices are rejected rather than silently producing incorrect output.
- Untagged input is assumed to be limited-range BT.709; test visually and provide correctly tagged SDR sources for non-BT.709 footage.
- If GPU encoding is unavailable, the app falls back to CPU-based x264 (8-bit) or x265 (10-bit)
- Select an existing **output folder** in either mode; manually typing an MP4 output filename remains supported in single-file mode.
- Batch mode processes video files directly inside the selected input folder; it does not scan subfolders.
- Batch mode skips files named `*_HDR.*` and `*_SDR.*` when the input and output folder are the same, to avoid reprocessing previous output. Existing destination files are still overwritten by FFmpeg (`-y`).
- If FFmpeg is not found, install it and add it to PATH; on macOS the standard Homebrew locations are also checked.
- Audio passthrough only works with compatible input formats and codecs
- FFmpeg output is streamed to the log viewer in real time
- Cancel safely terminates long-running conversions
- HDR preview is not yet implemented. Use an HDR-capable player/display to verify the resulting picture.

### Run the regression tests

```bash
python -m unittest discover -s tests -v
```

The GUI integration tests need the application's PyQt5, NumPy and OpenCV dependencies. The optional FFmpeg integration tests also exercise the actual zscale filter against synthetic skin-like and neutral patches, including a PQ-to-SDR round trip. GitHub Actions runs the GUI tests in an offscreen Qt environment.

---

## Technical reference

[MovieLabs: Best Practices for Mapping BT.709 Content to HDR10 (PDF)](https://www.movielabs.com/ngvideo/MovieLabs_Mapping_BT.709_to_HDR10_v1.0.pdf). Its 200-nit SDR-reference-white, linear display-light, BT.2020 primary conversion and PQ method are used here. FFmpeg's [zscale filter documentation](https://ffmpeg.org/ffmpeg-filters.html#zscale-1) documents the filter parameters.

**Known limitation:** this conversion preserves SDR colour/detail within an HDR transfer function; it is **not** AI reconstruction of original clipped highlights and not an artistically graded native HDR master.

---

## 📄 License

MIT License

---

## 🙋‍♂️ Author

Developed by Coolythecoder

---

## 🙌 Acknowledgments

- [FFmpeg](https://ffmpeg.org/)
- [x264](https://www.videolan.org/developers/x264.html)
- [x265](https://x265.org/)
- [PyQt5](https://riverbankcomputing.com/software/pyqt/)
