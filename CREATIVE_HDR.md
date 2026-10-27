# Creative HDR grading and luminance expansion

This optional mode adds a creative grade to both standard **10-bit HDR** and **YouTube HDR** export. It is **disabled by default**. Leaving the group unchecked retains the existing accurate conversion, filter arguments and audio behaviour.

## Use

1. Open the original SDR input and select an output destination as usual.
2. Check **Creative HDR grade (optional)**.
3. Start with **Gentle 400 nits** or **Creative 600 nits**; adjust the controls to taste.
4. Export a short test through either the standard 10-bit route or the YouTube button.
5. Inspect it on an HDR-capable display. **Reset to accurate conversion** unchecks the group and restores its controls. There is no native live HDR preview in this application.

A new executable must be built from the updated source; a previously released EXE does not acquire this feature automatically.

## Controls

| Control | Meaning |
| --- | --- |
| Peak ceiling | 200–4,000 nits. Must be at least reference white. A safety ceiling, not a promise that content reaches that value. |
| Highlight expansion strength | 0–1. Expands pixels above the luminance knee without lifting lower tones when the other controls are neutral. |
| Highlight start | 0.50–0.95 of **linear-light** reference white; not an SDR gamma-encoded slider percentage. |
| Roll-off softness | Exponential shoulder approaching the peak smoothly. |
| Exposure | -2 to +2 stops. Global brightness adjustment. |
| Contrast | 0.7–1.4, pivoted at 18% linear grey. |
| Midtone balance | -1 to +1 stop, weighted around middle grey. |
| Shadows | -2 to +2 stops, weighted toward dark tones; preserves zero black. |
| Black lift | Explicit 0–2 nit lift near black. Default zero. |
| Saturation | 0–1.5. Zero produces monochrome. |
| Vibrance | -0.5–1.0; affects low-saturation colours more than already saturated colours. Not a skin-recognition filter. |
| Temperature / tint | Creative warm/cool and green/magenta gains, -1–1. These are **not calibrated Kelvin** adjustments. |

Presets do not automatically increase contrast or saturation:

- Gentle: 400-nit ceiling, 0.35 expansion strength.
- Creative: 600-nit ceiling, 0.50 expansion strength.
- Bright: 1,000-nit ceiling, 0.60 expansion strength.

At the 200-nit reference-white default, the Creative preset maps a neutral source-white patch to approximately **400 nits**, while ordinary midtones remain unchanged. The ceiling allows additional headroom for exposure, stronger expansion and source excursions. A 600-nit ceiling does not mean every white patch becomes 600 nits.

## Colour pipeline

```
Declared SDR matrix/range/transfer/primaries
    -> linear-light, float32 BT.2020 RGB
    -> exposure and creative white balance
    -> luminance-based contrast/midtone/shadow adjustments
    -> saturation/vibrance
    -> highlight expansion, soft shoulder and gamut compression
    -> PQ encoding and 10-bit limited-range YUV
```

FFmpeg `geq` works on floating-point RGB here, not 8-bit SDR samples. The generic expressions use only validated numeric controls, without external scripts, downloaded LUTs or new runtime dependencies. RGB gain normalization preserves neutral-patch luminance during temperature/tint adjustment. Luminance uses BT.2020 coefficients 0.2627 R + 0.6780 G + 0.0593 B.

Highlight expansion is a deterministic, per-pixel curve. It does not change from frame to frame based on scene statistics. Above the knee, a quadratic term raises luminance. An exponential shoulder bounds it smoothly. Colours are compressed toward the corresponding neutral at the same luminance to fit the target RGB volume, rather than independently clipping saturated channels. This preserves the direction of linear RGB chroma; it is not a guarantee of perceptually constant hue at all settings.

**Full colour support** means all RGB channels are graded in a floating-point BT.2020 working space and encoded with the existing PQ/BT.2020 signalling. It does **not** mean changing limited-range video into full-range video, accepting every possible source colour space, or recovering authentic wide-gamut colours absent from the original. Existing supported SDR input profiles/ranges are retained. The YouTube preset still requires SDR BT.709 input. Already-HDR PQ/HLG sources remain rejected.

## Metadata and export safeguards

- Creative mode requires FFprobe and FFmpeg `geq`, `zscale`, and `format` filters.
- Standard 8-bit export is rejected while creative grading is enabled, rather than producing 8-bit video labelled HDR.
- **All creative exports perform the full content-light scan after the grade.** The scan and encoder use the identical linear-stage filter string and an immutable snapshot of the controls.
- Manual MaxCLL/MaxFALL overrides are disabled/ignored in creative mode, because a previous measurement may no longer describe the regraded image. Normal accurate mode preserves its existing override behaviour.
- CPU x265 uses BT.2020/D65 **virtual mastering-target** metadata and the selected peak ceiling. It is not a measurement of a physical mastering monitor. Content-light values start as unknown, then are populated by the scan before encoding.
- The existing analysis measures peak CIE luminance and maximum frame-average luminance before lossy encoding; compression and 10-bit quantization can slightly alter final pixel extremes. This is not a certification of a physical display or a professional grading workflow.
- Standard NVENC can use the creative pixel pipeline, but this application's NVENC path still does not embed x265-specific static mastering SEI. Its measurements are logged; use CPU x265 / YouTube export for embedded mastering and content-light metadata.
- Quality, audio, frame-rate and container handling are otherwise unchanged. Source files are not changed.
- Float `geq` evaluation adds CPU work to both the scan and encoding pass, especially at 4K60. It is not a GPU-accelerated colour grader.

## Validation

The tests cover disabled-mode argument and pixel identity, parameter validation, monotonic neutral ramps, black preservation, brighter highlights, finite/bounded saturated colours, temperature/tint, monochrome saturation, full/limited source range equivalence, immutable GUI settings and forced post-grade analysis. Real FFmpeg integration tests encode an HDR10 MP4, inspect its measured light-level and mastering SEI, and decode PQ samples back to linear light to verify that the pixels—not merely the tags—became brighter.

Windows/macOS GUI and unit tests do not substitute for real-device HDR viewing or hardware NVENC testing. Test your own footage, particularly bright text, flat white animation areas and specular highlights: a global luminance threshold cannot distinguish these semantically.

## References

- FFmpeg `geq` and `zscale`: https://ffmpeg.org/ffmpeg-filters.html
- x265 mastering-display/content-light SEI: https://x265.readthedocs.io/en/master/cli.html

These references describe the filter and metadata interfaces. The creative tone curve is this application's implementation, not a claim of conformance to a standardized inverse-tone-mapping operator.
