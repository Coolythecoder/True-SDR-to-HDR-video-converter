# Live grading workspace

## Use

1. Open your **SDR source/intermediate** at the top (or drop a local video onto the window). Its preview loads automatically. An already-PQ/HLG video is rejected, as in the exporter.
2. Choose a representative frame with the timeline, seconds box, or frame-step buttons.
3. Enable **Creative HDR grade**, select a preset, then use the **Highlights**, **Tone** and **Colour** tabs. Each slider is linked to a precise numeric value and has an individual reset button. Preset changes apply as one preview update.
4. Use **Split comparison** and drag the vertical divider. **Before grade** means the accurate HDR mapping without the creative grade; **After grade** uses the current grade. Both use the same SDR viewing transform.
5. Choose **Export settings** to configure YouTube or custom export. The two export buttons and Cancel/status remain visible below the preview. Export settings are separate; normal custom CRF/GPU/audio controls do not override the YouTube preset.

**Play proxy** shows silent, reduced-temporal-resolution previews, advancing by a quarter-second of source time per rendered result. It is intended for browsing the grade, not checking 60 fps motion, interpolation or A/V sync. Performance is hardware/source dependent. Frame stepping is based on the reported average source frame rate; variable-rate clips can seek to the nearest decoded frame rather than an exact editorial frame index.

## What the view does—and does not—show

This is a **live SDR tone-mapped approximation**, not a native HDR swapchain or a reference mastering display. The app reuses the exporter's `build_creative_filter` expressions on float32 linear BT.2020 RGB. The viewing stage then uses a fixed 1000-nit Mobius shoulder and converts into sRGB for a normal Qt image surface. Cached raw frames are explicitly tagged as linear BT.2020 before the viewing transform.

The viewing scale is shared by both halves and does not automatically normalize to the currently selected creative peak; changing highlight expansion remains visible. White/colour controls change actual preview pixels. Values above the 1000-nit display scale and colours outside sRGB can clip in this preview even when the HDR export retains them. It cannot display real 600/1000-nit brightness or the complete BT.2020 colour volume. Inspect an actual export on an HDR-capable player/display to judge those properties.

The preview resizes **before** grading for interactivity. It is not a full-resolution pixel proof: tiny specular highlights, chroma edges, dithering, HEVC compression and measured content-light extrema can differ from the full-resolution output. Preview resolution, split position, timeline position and playback do not alter the export. Exports still process the full source at its original dimensions/timing with post-grade metadata analysis.

## Responsiveness and safety

- Metadata probing, frame decoding and rendering are asynchronous `QProcess` jobs; ordinary preview requests never wait for FFmpeg on the UI thread.
- Control changes are debounced for 160 ms. Only the most recent result is displayed; superseded render jobs are cancelled.
- Up to three decoded proxy frames are cached. Regrading the same frame reuses its decoded float samples. Source changes clear the cache and image immediately.
- **Fast 480**, **Balanced 640** (default) and **Detail 960** fit inside 16:9 proxy boxes while retaining source display aspect, including portrait/anamorphic and rotation metadata.
- Batch mode never sends a directory to FFmpeg. Use **Choose preview clip** to inspect a representative file; this does not change the batch input.
- Missing tools, unsupported colour tags, malformed/incomplete output and timeouts produce visible errors. Preview buffers are size-limited and stderr is bounded.
- Preview processing pauses during export. Grade/source controls are disabled while the exporter uses its snapshot; the log remains accessible.
- Closing stops preview processes. Closing during export asks for cancellation and waits for the worker to finish before destroying the window.
- Source videos are not modified and no preview file is written beside the source.

## Dependencies and packaging

No new Python runtime dependency is introduced: PyQt5, NumPy and OpenCV remain the application dependencies. FFmpeg must include `zscale`, `geq`, `tonemap`, `setparams`, `split`, `hstack` and `format`; FFprobe is required for the preview. Missing filter errors appear in the preview status rather than changing the video silently.

Run the same `SDR_to_HDR_cross_platform.py` entry point. Rebuild the EXE from the updated repository so PyInstaller includes `grading_preview.py`, `preview_widget.py` and `workspace_ui.py` along with the existing modules. Previously published EXEs do not update automatically.

## Validation

New tests cover source colour/range metadata, rotation/aspect geometry, shared grading expressions, disabled-grade pixel identity, separate 200/400/600/1000-nit highlight patches, saturation, warm/cool adjustments, async cancellation, stale-result rejection, frame caching, batch isolation and export/preview separation. A real Qt/FFmpeg integration test opens a Unicode filename, renders, regrades from cache, seeks and shuts down; Linux CI captures actual workspace screenshots. Platform skips are reported explicitly when FFmpeg is unavailable.

References: FFmpeg filter documentation https://ffmpeg.org/ffmpeg-filters.html and Qt process/image APIs https://doc.qt.io/qt-6/qprocess.html . The implementation uses the PyQt5-compatible API subset; it does not require Qt 6.
