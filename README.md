# SDR to HDR Video Converter (Cross-Platform)

A Python + PyQt5 desktop application to convert SDR (Standard Dynamic Range) videos to HDR (High Dynamic Range) using FFmpeg with support for x264/x265 encoding. Offers tone mapping, HDR metadata embedding, color space conversion, batch conversion, and real-time preview.

---

## 🚀 Features

- ✅ **SDR to HDR video conversion** using FFmpeg
- 🎛️ **Tone mapping modes**: Linear, Logarithmic, PQ (Perceptual Quantizer)
- 🔢 **Custom controls** for tone mapping parameters (gamma, scale, compression)
- 🎚️ **Constant Quality (CRF) slider**
- 🌈 **Color space conversion**: BT.709 to BT.2020
- 📝 **Static HDR10 metadata embedding** (MaxCLL, MaxFALL)
- ⚙️ **Metadata estimation** or **manual override**
- 📦 **Batch conversion mode** (process entire folders)
- 👁️ **Real-time preview** of tone mapping
- 🖱️ **Drag and drop support** for video files and folders
- ❌ **Graceful cross-platform cancellation** of FFmpeg subprocesses

---

## 🖥️ GUI Overview

- **Input/Output**: Choose files or folders for conversion and output
- **Tone Mapping**:
  - Mode selection (Linear, Log, PQ)
  - Dynamic controls per mode (e.g., PQ gamma, Log factor, Linear scale)
- **Encoding Settings**:
  - Output bit depth: 8 or 10-bit
  - Constant Quality (CRF) slider
- **Metadata Settings**:
  - Embed HDR10
  - Estimate from video
  - Manually override MaxCLL/MaxFALL
- **Modes**:
  - Batch folder mode
  - Real-time preview mode
- **Cancel Button**: With confirmation prompt and safe termination

---

## 🛠️ Requirements

- **Python 3.7+**
- **FFmpeg** (must be in system PATH and support libx264 or libx265)
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

To build a Windows executable:

```bash
pip install pyinstaller
pyinstaller --onefile --windowed SDR_to_HDR_cross_platform.py
```

> Executable will be created in the `dist/` folder.

---

## 📂 Supported Input Formats

- `.mp4`
- `.mov`
- `.mkv`

---

## ⚠️ Notes

- Metadata estimation uses the first 30 frames for luminance analysis.
- Drag-and-drop accepts both individual files and directories (for batch mode).
- Cancel button uses `CTRL_BREAK_EVENT` on Windows and `terminate()` on Unix-like systems.
- Real-time preview opens a window via OpenCV; press `q` or close it to exit preview.

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
