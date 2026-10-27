# SDR to HDR Video Converter (Cross-Platform)

A Python + PyQt5 desktop application to convert SDR (Standard Dynamic Range) videos to HDR (High Dynamic Range) using FFmpeg. Supports advanced tone mapping, GPU acceleration, HDR10 metadata embedding, batch mode, **optional audio passthrough**, and real-time logging.

---

## 🚀 Features

- ✅ **SDR to HDR video conversion** using FFmpeg
- ⚡ **GPU acceleration** with NVENC (with auto fallback to CPU/x265)
- 🧠 **Auto encoder switching** based on FFmpeg availability
- 🎛️ **Tone mapping modes**: Linear, Logarithmic, PQ (Perceptual Quantizer)
- 🔢 **Custom tone mapping controls** (gamma, scale, compression)
- 🎚️ **Constant Quality (CRF) slider**
- 🌈 **Color space conversion**: BT.709 to BT.2020
- 🔊 **Audio passthrough toggle**: retain original audio or re-encode to AAC
- 📊 **HDR10 metadata**: auto-generate or manually override `MaxCLL`, `MaxFALL`, `master-display`
- 📝 **`-x265-params`** integration for HDR metadata (compatible with libx265)
- 🖥️ **Real-time FFmpeg log viewer**
- 📦 **Batch conversion mode** (process MP4/MOV/MKV files in a folder, one by one)
- 👁️ **Real-time tone mapping preview**
- 🖱️ **Drag and drop support** for video files and directories
- ❌ **Safe cancellation** of conversions (Windows & Linux)

---

## 🖥️ GUI Overview

- **Input/Output**: Select a video (or folder in batch mode) and an existing output folder. Output filenames are generated as `<input>_HDR.mp4`.
- **Tone Mapping**: Choose Linear, PQ, or Log; each with dynamic controls
- **Encoding Options**:
  - Bit depth: 8-bit or 10-bit
  - CRF slider (0–51)
  - GPU toggle (NVENC)
  - **Audio Passthrough**: Optional toggle to copy original audio stream
- **HDR Metadata Options**:
  - Estimate or override MaxCLL/MaxFALL
  - Embed static HDR10 metadata
- **Log Output**: Real-time FFmpeg log monitor
- **Execution**: Batch mode, preview mode, and a cancel-safe exit

---

## 🛠️ Requirements

- **Python 3.7+**
- **FFmpeg** (in PATH, with libx265, nvenc, or libx264 support)
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

- HDR metadata is only embedded when using `libx265` or `hevc_nvenc`
- If GPU encoding is unavailable, the app falls back to CPU-based x264 (8-bit) or x265 (10-bit)
- Select an existing **output folder** in either mode; manually typing an MP4 output filename remains supported in single-file mode.
- Batch mode processes video files directly inside the selected input folder; it does not scan subfolders.
- Batch mode skips files named `*_HDR.*` when the input and output folder are the same, to avoid reprocessing previous output. Existing destination files are still overwritten by FFmpeg (`-y`).
- If FFmpeg is not found, install it and add it to PATH; on macOS the standard Homebrew locations are also checked.
- Audio passthrough only works with compatible input formats and codecs
- FFmpeg output is streamed to the log viewer in real time
- Cancel safely terminates long-running conversions
- Preview mode uses OpenCV window (press `q` to exit)

### Run the regression tests

```bash
python -m unittest discover -s tests -v
```

The GUI integration tests need the application's PyQt5, NumPy and OpenCV dependencies; GitHub Actions installs these and runs in an offscreen Qt environment.

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
