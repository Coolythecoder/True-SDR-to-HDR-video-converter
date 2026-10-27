# SDR to HDR Video Converter (Cross-Platform)

A Python + PyQt5 desktop application to map SDR video into 10-bit BT.2020/PQ output using FFmpeg. The **HDR Video Studio** interface provides a live grading preview, optional creative highlight expansion, CPU or NVENC encoding, batch conversion, and a verified YouTube HDR export preset.

This is real pixel conversion, not just HDR metadata tagging. Creative grading can expand highlight brightness, but cannot recover detail already clipped in the original SDR recording.

## Preview-first workflow

1. **Source:** open your SDR video/intermediate and choose an output folder. The preview loads automatically. A local video can also be dragged onto the window.
2. **Preview and grade:** choose a frame with the timeline. Enable **Creative HDR grade**, select a preset and adjust the **Highlights**, **Tone** or **Colour** controls. Sliders and precise numeric values are linked; each control has a reset button.
3. **Compare:** drag the divider in **Split comparison**, or select **Before grade** / **After grade**. Grade changes update without encoding the whole video.
4. **Export:** select **Export settings**, then use the persistent **Export YouTube HDR** or **Export custom** button. Detailed processing messages are available under **Log**.

**The preview is an SDR tone-mapped approximation, not native HDR monitor output.** It shares the exporter's float-linear grading expressions, but maps the result into sRGB for a normal Qt image surface. Both halves use the same fixed viewing transform so highlight changes stay visible. Actual HDR brightness and colours outside sRGB require checking an export on an HDR display. Preview resolution and playback do not change export settings.

See [Live preview guide](LIVE_PREVIEW.md) for display limits, timeline controls, proxy playback and batch previews, and [Creative HDR guide](CREATIVE_HDR.md) for grading mathematics and metadata behaviour.

## Features

- Accurate source SDR transfer/range/primaries conversion into float32 linear BT.2020, followed by 10-bit PQ encoding.
- Adjustable HDR reference white, 80–400 nits; default 200 nits.
- Optional creative HDR grading: highlight expansion and soft roll-off, exposure, contrast, midtones, shadows, black lift, saturation, vibrance, temperature and tint.
- Live before/after preview with a draggable divider, source-frame cache, timeline, frame stepping and silent proxy playback. Processing runs asynchronously and outdated results are discarded.
- Preview quality choices: Fast 480, Balanced 640 and Detail 960. These are preview sizes, not export resolutions.
- CPU x264/x265 or optional NVENC encoding in custom mode; automatic fallback when the requested encoder is absent from the FFmpeg build.
- YouTube HDR preset: CPU HEVC Main 10, BT.2020/PQ, static HDR10 metadata, adjustable quality, measured content-light values, optional two-pass audio normalization, AAC 48 kHz stereo and MP4 fast-start.
- Batch MP4/MOV/MKV conversion, progress logging and cancellation. Batch preview uses a separately selected representative clip; folders are not decoded as videos.
- Separate Grade / Export settings / Log tabs, resizable preview split, collapsible advanced metadata and always-visible export controls.

## Requirements

- Python 3.7+ syntax compatibility; CI uses Python 3.11. Use a Python version supported by your installed dependencies.
- `PyQt5`, `numpy`, `opencv-python`.
- **FFmpeg** on PATH, including `zscale`/libzimg and the appropriate encoder. Creative grading requires `geq`; preview additionally requires `tonemap`, `setparams`, `split`, `hstack` and `format`.
- **FFprobe** is required for live preview, creative grading and the YouTube preset. Standard accurate conversion can fall back to an explicitly logged BT.709 limited-range assumption without it.
- **libx265** is required for YouTube export. That preset deliberately does not use NVENC because the application's NVENC path does not embed x265-specific static mastering SEI.

```bash
pip install PyQt5 numpy opencv-python
python SDR_to_HDR_cross_platform.py
```

Most complete FFmpeg distributions include FFprobe and the required filters. On macOS the application also checks common Homebrew installation paths.

## Build the Windows executable

From the complete updated repository:

```powershell
python -m pip install pyinstaller
python -m PyInstaller --clean --onefile --windowed SDR_to_HDR_cross_platform.py
```

The executable is generated in `dist`. The imports include the preview/workspace modules automatically. **Previously published EXEs do not acquire source changes automatically**; rebuild before testing the new interface. FFmpeg/FFprobe remain external tools and are not bundled by this command.

## Export modes

### Custom export

**Export custom** uses the controls in the Custom export section: 10-bit HDR or 8-bit SDR, CPU/NVENC, constant quality, and optional audio copy. Lower CRF means a higher quality target; hardware encoding uses its own quantizer setting.

10-bit output uses `_HDR.mp4`; 8-bit output uses `_SDR.mp4` and is not labelled HDR. Creative grading is rejected for 8-bit output. Audio copy requires the source codec to be compatible with MP4.

CPU x265 can embed static ST 2086 mastering and content-light metadata. NVENC produces PQ/BT.2020 signalling but this application's GPU path does not embed those static x265 SEI fields. The creative pixel grade itself remains CPU-based on either route.

### YouTube HDR export

Choose a quality profile under **Export settings → YouTube HDR export**, leave **Measure content-light metadata** enabled unless supplying known values in accurate mode, optionally enable **Normalize audio**, and click **Export YouTube HDR**.

The preset converts SDR to 10-bit BT.2020/PQ and encodes HEVC Main 10 with CPU x265. It writes HDR10 mastering and content-light metadata, first-audio-stream stereo AAC at 48 kHz / 384 kb/s, `hvc1`, MP4 fast-start and no edit lists. Output filenames end in `_YouTube_HDR.mp4`.

After encoding, FFprobe checks the actual video profile, pixel format, colour signalling, static metadata and audio characteristics; container checks verify fast-start and the absence of edit lists. An output that fails verification is not reported as successfully verified. These checks cannot guarantee YouTube ingestion or visual quality.

| YouTube quality | x265 CRF | Speed preset |
| --- | --- | --- |
| Balanced | 20 | medium |
| High (default) | 16 | medium |
| Maximum | 12 | slow |

These are quality-based variable-bitrate profiles, not mandatory bitrate targets. The normal custom quality/GPU/audio-copy controls do not override the YouTube preset.

**Source restrictions:** supported SDR BT.709 primaries, even dimensions, at least 720 pixels on the short edge, and common frame rates (23.976/24/25/29.97/30/48/50/59.94/60 fps). Declared interlaced footage and unsupported sizes/rates are rejected by this preset. Input dimensions and timing are retained rather than silently upscaled or interpolated. Already-HDR PQ/HLG sources are rejected by the SDR converter.

**Automatic light analysis:** a full-resolution pass measures peak pixel CIE luminance and highest frame-average luminance in the display-linear conversion pipeline, and writes the calculated MaxCLL/MaxFALL fields. Lossy encoding can cause small differences in final pixel extrema. Accurate mode honours explicit manual overrides; with analysis disabled its defaults are reference-white MaxCLL (normally 200 nits) and unknown MaxFALL (0).

**Creative mode always analyses after grading**, using exactly the same linear-stage grade as the encoder. Previous manual values are not reused. Its mastering metadata describes a virtual BT.2020/D65 target and selected peak ceiling, not a measurement of a physical mastering monitor.

**Audio normalization** is optional and off by default. It measures the first audio stream after the same stereo downmix used for export, then applies a measured second-pass FFmpeg loudnorm filter targeting −16 LUFS, −1.5 dBTP and LRA 11. The target is an application choice, not an official YouTube requirement. FFmpeg may use dynamic normalization when linear gain cannot meet the constraints; dynamics can change. Silent/no-audio sources are left unnormalized. AAC encoding can slightly change decoded peaks.

Upload the completed file through YouTube Studio normally; the app does not upload or authenticate to YouTube. Use footage you have permission to upload and check a short unlisted test after processing.

- [YouTube HDR upload requirements](https://support.google.com/youtube/answer/7126552?hl=en)
- [YouTube upload encoding settings](https://support.google.com/youtube/answer/1722171?hl=en)

## Preview and grading limits

- **Before grade** is the accurate conversion without the creative treatment, viewed through the same SDR display transform as **After grade**. It is not a native HDR presentation or an untouched-source player.
- The preview resizes before grading for responsiveness; it is not a full-resolution pixel proof. Export still uses full-resolution pixels and content-light analysis.
- **Play proxy** is a silent, performance-limited frame browser, not smooth 60 fps playback or an audio-sync check. The timeline and frame buttons are for choosing representative grading frames.
- Preview pauses while an export is running; source and grading controls are locked to prevent confusion with the export snapshot. The log stays accessible.
- Creative colour controls are not recovery of authentic wide-gamut detail, and the peak ceiling is not a promise that every white object reaches that luminance.
- Accurate mode remains the default. Reset or uncheck Creative HDR grade to restore it.

## File and batch behaviour

Supported inputs are `.mp4`, `.mov` and `.mkv`. Select an existing output folder; an explicit `.mp4` destination remains supported in single-file mode. Source files are not modified.

Batch mode processes files directly in the input folder, not subfolders. When input/output folders are the same it skips files with generated `_HDR`, `_SDR` and `_YouTube_HDR` endings. **Existing destination files are overwritten** by the existing FFmpeg `-y` behaviour; use a different output folder when retaining earlier grades.

Supported source colour metadata is honoured. Untagged sources assume limited-range BT.709, disclosed in the preview or logs as applicable. Other declared unsupported colour profiles are rejected rather than silently misinterpreted.

## Tests

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs GUI/import/unit tests on Windows, Linux and macOS. FFmpeg-dependent tests run where the tools are installed (Linux CI installs them). Tests cover existing export behaviour, colour fidelity, grading, metadata, audio, asynchronous preview cancellation/stale results and an actual Qt/FFmpeg live-preview session. Linux CI stores rendered workspace screenshots as an artifact. Skips are reported rather than counted as media-pipeline validation.

## References and limitations

- [MovieLabs BT.709-to-HDR10 mapping](https://www.movielabs.com/ngvideo/MovieLabs_Mapping_BT.709_to_HDR10_v1.0.pdf)
- [FFmpeg filter documentation](https://ffmpeg.org/ffmpeg-filters.html)
- [x265 documentation](https://x265.readthedocs.io/en/master/cli.html)
- [PyQt5](https://riverbankcomputing.com/software/pyqt/)

This is a conversion and creative regrading tool, not reconstruction of clipped source highlights or certification of a professional HDR master. Actual HDR display review remains necessary.

## License and author

MIT License. Developed by Coolythecoder.
