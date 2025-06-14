# SDR to HDR Video Converter

A desktop application built with Python and PyQt5 that allows users to convert SDR (Standard Dynamic Range) videos into HDR (High Dynamic Range) format using FFmpeg with x265 encoding. Supports tone mapping, color space conversion, real-time preview, batch processing, and optional embedding of HDR10 static metadata.

---

## 🚀 Features

- ✅ **SDR to HDR video conversion using FFmpeg (x265)**
- 🎚️ **Tone mapping options**: Linear, Logarithmic, PQ (Perceptual Quantizer)
- 🌈 **Color space conversion**: BT.709 to BT.2020
- 📦 **Batch mode**: Convert entire folders of videos
- 👁 **Real-time preview** of tone mapping effects
- 📝 **Embed static HDR10 metadata**, including MaxCLL and MaxFALL
- ✏️ **Manual or automatic metadata generation**
- 🖱️ **Drag-and-drop interface**
- 🛑 **Graceful cancellation** of ongoing conversions

---

## 🖥️ GUI Overview

- **Input Path**: Choose one or more video files or a folder
- **Output Path**: Choose save location or output directory
- **Tone Mapping**: Select the type of tone mapping
- **Bit Depth**: 8-bit or 10-bit encoding
- **Color Conversion**: Option to convert to BT.2020 color space
- **Metadata Options**:
  - Embed HDR10 metadata
  - Generate static metadata (with override option)
- **Modes**:
  - Batch processing
  - Real-time preview
- **Cancel Button**: Interrupts FFmpeg safely and updates status accordingly

---

## 🛠️ Requirements

- **Python 3.7+**
- **FFmpeg** with `libx265` support installed and accessible in system PATH
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
python sdr_to_hdr_converter.py
```

---

## 🏗️ Build Executable (Optional)

To create a standalone Windows executable:

```bash
pip install pyinstaller
pyinstaller --onefile --windowed sdr_to_hdr_converter.py
```

Executable will be in the `dist/` folder.

---

## 📂 Supported Formats

- `.mp4`
- `.mov`
- `.mkv`

---

## 📝 Notes

- HDR metadata generation uses the first 30 frames of the video to estimate luminance.
- Canceling a conversion will **gracefully terminate** the active FFmpeg process on both Windows and Unix platforms.
- Tone mapping is previewed using OpenCV's imshow (close the window or press `q` to exit preview).

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙋‍♂️ Author

Developed by Coolythecoder

---

## ❤️ Acknowledgments

- [FFmpeg](https://ffmpeg.org/)
- [x265](https://x265.org/)
- [PyQt5](https://riverbankcomputing.com/software/pyqt/)
