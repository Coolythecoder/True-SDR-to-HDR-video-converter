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
- ▶️ **YouTube HDR upload preset**: HEVC Main 10, BT.2020/PQ, x265 HDR10 metadata, adjustable quality presets, measured MaxCLL/MaxFALL, optional two-pass audio loudness normalization, AAC 48 kHz, MP4 fast-start and automatic export validation
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
- **YouTube HDR button**: Separate upload preset. Uses CPU/libx265 and 10-bit PQ with selectable Balanced/High/Maximum quality, automatic full-frame light-level analysis, and optional EBU R128 loudness normalization. Normal conversion settings are unchanged.
- **Log Output**: Real-time FFmpeg log monitor
- **Execution**: Batch mode, preview mode, and a cancel-safe exit

---

## 🛠️ Requirements

- **Python 3.7+**
- **FFmpeg** (in PATH, with the **zscale/libzimg filter** and libx265 or hevc_nvenc for HDR export)
- **FFprobe** (recommended for standard export, **required for YouTube HDR** to validate the source and encoded output).
- **YouTube HDR preset additionally requires a libx265-enabled FFmpeg**. NVENC is deliberately not used for this preset because the application's GPU path does not embed the static HDR10 mastering-display SEI.
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
- 10-bit HDR outputs use `_HDR.mp4`. 8-bit SDR outputs use `_SDR.mp4` and are not tagged as HDR. YouTube HDR outputs use `_YouTube_HDR.mp4`.
- HDR PQ reference white defaults to 200 nits: the original SDR image is mapped into an HDR container without inventing detail above the SDR source's available range.
- CPU/libx265 can embed static ST 2086 + MaxCLL/MaxFALL SEI; NVENC output is tagged PQ/BT.2020 but does **not** embed those x265-specific SEI fields.
- Only tested SDR colour formats can be converted. Declared HDR PQ/HLG inputs and unsupported declared primaries/transfers/matrices are rejected rather than silently producing incorrect output.
- Untagged input is assumed to be limited-range BT.709; test visually and provide correctly tagged SDR sources for non-BT.709 footage.
- If GPU encoding is unavailable, the app falls back to CPU-based x264 (8-bit) or x265 (10-bit)
- Select an existing **output folder** in either mode; manually typing an MP4 output filename remains supported in single-file mode.
- Batch mode processes video files directly inside the selected input folder; it does not scan subfolders.
- Batch mode skips files named `*_HDR.*`, `*_SDR.*` and `*_YouTube_HDR.*` when the input and output folder are the same, to avoid reprocessing previous output. Existing destination files are still overwritten by FFmpeg (`-y`).
- If FFmpeg is not found, install it and add it to PATH; on macOS the standard Homebrew locations are also checked.
- Audio passthrough only works with compatible input formats and codecs
- FFmpeg output is streamed to the log viewer in real time
- Cancel safely terminates long-running conversions
- HDR preview is not yet implemented. Use an HDR-capable player/display to verify the resulting picture.

### YouTube HDR export

1. Select an SDR video and an existing output folder (or use batch mode).
2. Choose a YouTube quality profile, leave **Measure MaxCLL / MaxFALL** enabled unless you have your own measured light-level metadata, and optionally enable **Normalize YouTube audio**. Click **Export YouTube HDR MP4**.
3. The preset converts SDR video into 10-bit Rec.2020/PQ using the existing colour-accurate zscale pipeline, encodes HEVC Main 10 with CPU/libx265 and embeds ST 2086 mastering + content-light metadata. It also transcodes the first audio stream to 48 kHz stereo AAC at 384 kb/s (downmixing multichannel audio) and writes an MP4 with the moov atom at the beginning and with MP4 edit lists disabled.
4. After FFmpeg finishes, FFprobe verifies the HEVC Main 10 profile, 10-bit format, PQ/Rec.2020 signalling, static HDR10 SEI, audio characteristics, and MP4 fast-start structure without edit lists. If validation fails, the app reports an error instead of success.
5. Upload the resulting `*_YouTube_HDR.mp4` through YouTube Studio normally. There is no automatic upload or OAuth integration.

**Source requirements:** progressive, even dimensions, at least 720 pixels on the shorter edge, common YouTube HDR frame rates (23.976/24/25/29.97/30/48/50/59.94/60 fps), tagged or correctly assumed SDR BT.709 primaries. Explicitly non-BT.709 sources, existing PQ/HLG HDR, interlaced footage and unsupported sizes/rates are rejected by this preset; use the standard converter for other material. The preset preserves input frame rate and dimensions rather than silently resampling.

**Quality profiles:** the YouTube exporter uses x265 quality-based, variable-bitrate encoding: **Balanced** = CRF 20/medium, **High** (default) = CRF 16/medium, and **Maximum** = CRF 12/slow. A smaller CRF produces higher quality/larger files; **there is no mandatory minimum bitrate**, because YouTube transcodes uploads and simple animations compress exceptionally well. This preset is not an HDR creative grade, and it cannot recover detail missing from SDR footage.

**Automatic HDR light metadata:** the full input is scanned in the same display-linear Rec.2020 stage used by the PQ encoder. The scan measures the brightest pixel's luminance (**MaxCLL**) and the brightest frame's average luminance (**MaxFALL**) in nits, then injects those numbers into the x265 content-light SEI for the export. No 8-bit greyscale guessing. This extra full-resolution pass can take substantial time on long/4K videos; it runs in the background conversion worker and can be cancelled. The measured values reflect the pre-encode mapping, so lossy HEVC compression can cause small output-level differences. If you explicitly check **Override HDR Metadata**, your manual values take precedence and automatic analysis is skipped. With analysis disabled and no override, defaults remain reference-white MaxCLL (normally 200 nits) and unknown MaxFALL (0).

**Optional audio normalization:** unchecked by default. When enabled and the input has audio, FFmpeg analyzes the entire first audio stream using EBU R128 loudnorm, with the same stereo downmix as the eventual output, then applies measured loudnorm settings during encoding. The target is **-16 LUFS integrated, at most -1.5 dBTP, LRA 11**. If linear correction cannot satisfy the true-peak/LRA constraints, FFmpeg can use dynamic normalization, which may alter dynamics. Audio is still 48 kHz stereo AAC. The -16 LUFS target is an **application choice, not an official YouTube requirement**. Silent/no-audio tracks are left unnormalized.

**Upload acceptance:** the exporter checks the local encoded file. YouTube still needs to ingest, analyse and transcode it; the HDR quality selection may not appear immediately or at all if YouTube rejects the metadata. Test a short unlisted upload before releasing a long video.

- [YouTube: upload HDR videos](https://support.google.com/youtube/answer/7126552?hl=en)
- [YouTube: recommended upload encoding settings](https://support.google.com/youtube/answer/1722171?hl=en)

### Run the regression tests

```bash
python -m unittest discover -s tests -v
```

The GUI integration tests need the application's PyQt5, NumPy and OpenCV dependencies. The optional FFmpeg integration tests also exercise the actual zscale filter against synthetic skin-like and neutral patches, full-frame HDR luminance metadata, and two-pass loudness normalization. GitHub Actions runs the GUI tests in an offscreen Qt environment.

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
