"""Optional pre-encode content-light and two-pass EBU R128 audio analysis.

Both passes run in the existing Qt conversion worker (never the GUI thread).
Colour analysis samples *every* decoded pixel/frame in display-linear BT.2020.
The mapping uses the same linear stage as the actual SDR-to-PQ conversion.
"""

import json
import math
import re
import subprocess
import tempfile

import numpy as np

from hdr_processing import build_sdr_to_linear_filter, x265_sdr_mastering_params

TARGET_LUFS = -16.0
TARGET_TRUE_PEAK = -1.5
TARGET_LRA = 11.0


class ExportCancelled(Exception):
    """An optional analysis pass was cancelled by the user."""


def _check_cancel(cancelled):
    if cancelled is not None and cancelled():
        raise ExportCancelled("Analysis cancelled.")


def linear_frame_light_levels(frame, white_nits):
    """Calculate peak and frame-average luminance in nits from GBR planar floats.

    FFmpeg gbrpf32le stores planes in G, B, R order, not R, G, B.
    Rec.2020 linear RGB -> CIE luminance: Y=0.2627 R + 0.6780 G + 0.0593 B.
    """
    if frame.ndim != 3 or frame.shape[0] != 3:
        raise ValueError("Expected three planar GBR channels.")
    if not np.isfinite(frame).all():
        raise ValueError("Non-finite luminance encountered in decoded video.")
    green, blue, red = frame[0], frame[1], frame[2]
    lum = (0.2627 * red + 0.6780 * green + 0.0593 * blue)
    np.clip(lum, 0.0, 10000.0 / white_nits, out=lum)
    return (
        float(lum.max()) * white_nits,
        float(lum.mean(dtype=np.float64)) * white_nits,
    )


def _read_frame(pipe, frame_bytes, cancelled=None):
    frame = bytearray(frame_bytes)
    view = memoryview(frame)
    position = 0
    while position < frame_bytes:
        _check_cancel(cancelled)
        chunk = pipe.readinto(view[position:])
        if not chunk:
            if position:
                raise ValueError("Partial raw video frame in brightness analysis.")
            return None
        position += chunk
    return frame


def measure_content_light(
    ffmpeg_exe, input_path, profile, width, height, white_nits,
    register_process=None, cancelled=None, progress=None, grade=None
):
    """Return (MaxCLL, MaxFALL, frame_count) after the optional creative grade.

    This is a pre-encode prediction in the same linear colour pipeline as the
    exported PQ video. Encoding may change extreme pixels slightly.
    """
    if width < 1 or height < 1 or width * height > 100000000:
        raise ValueError("Invalid dimensions for luminance measurement.")
    if not 80 <= white_nits <= 400:
        raise ValueError("Invalid HDR reference white.")
    frame_bytes = width * height * 3 * 4
    command = [
        ffmpeg_exe, "-hide_banner", "-nostdin", "-nostats", "-v", "error",
        "-noautorotate", "-i", str(input_path),
        "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", build_sdr_to_linear_filter(profile, grade=grade, white_nits=white_nits),
        "-pix_fmt", "gbrpf32le", "-f", "rawvideo", "pipe:1",
    ]
    _check_cancel(cancelled)
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as errors:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=errors, bufsize=1024 * 1024
        )
        if register_process is not None:
            register_process(process)
        max_pixel, max_frame = 0.0, 0.0
        frame_count = 0
        try:
            _check_cancel(cancelled)
            while True:
                frame_data = _read_frame(process.stdout, frame_bytes, cancelled)
                if frame_data is None:
                    break
                planar = np.frombuffer(frame_data, dtype="<f4").reshape(
                    (3, height, width)
                )
                pixel, frame_average = linear_frame_light_levels(planar, white_nits)
                max_pixel = max(max_pixel, pixel)
                max_frame = max(max_frame, frame_average)
                frame_count += 1
                if progress is not None and frame_count % 120 == 0:
                    progress("Analysed {} video frames for HDR light levels.".format(
                        frame_count
                    ))
            returncode = process.wait()
            _check_cancel(cancelled)
            if returncode:
                errors.seek(0)
                raise ValueError("FFmpeg HDR light analysis failed: {}".format(
                    errors.read()[-2000:]
                ))
            if not frame_count:
                raise ValueError("Cannot calculate HDR metadata from an empty video.")
            max_cll = max(1, min(10000, int(math.ceil(max_pixel))))
            max_fall = max(1, min(max_cll, int(math.ceil(max_frame))))
            return max_cll, max_fall, frame_count
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if process.stdout:
                process.stdout.close()
            if register_process is not None:
                register_process(None)


def _loudnorm_filter(measured=None, print_format="none"):
    parameters = [
        "I={:g}".format(TARGET_LUFS),
        "TP={:g}".format(TARGET_TRUE_PEAK),
        "LRA={:g}".format(TARGET_LRA),
    ]
    if measured is not None:
        fields = {
            "input_i": "measured_I",
            "input_lra": "measured_LRA",
            "input_tp": "measured_TP",
            "input_thresh": "measured_thresh",
            "target_offset": "offset",
        }
        for source_key, target_key in fields.items():
            value = float(measured[source_key])
            if not math.isfinite(value) or value < -99 or value > 99:
                raise ValueError("Invalid loudness statistic: {}".format(source_key))
            parameters.append("{}={:g}".format(target_key, value))
        parameters.append("linear=true")
    parameters.append("print_format={}".format(print_format))
    # Both analysis and encoding downmix first, to ensure matched loudness.
    return "aformat=channel_layouts=stereo,loudnorm=" + ":".join(parameters)


def measure_audio_loudness(
    ffmpeg_exe, input_path, register_process=None, cancelled=None
):
    """Return statistics for a second-pass loudnorm or None for silent input."""
    _check_cancel(cancelled)
    command = [
        ffmpeg_exe, "-hide_banner", "-nostdin", "-nostats", "-v", "info",
        "-i", str(input_path), "-map", "0:a:0", "-vn", "-sn", "-dn",
        "-af", _loudnorm_filter(print_format="json"),
        "-f", "null", "-",
    ]
    process = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if register_process is not None:
        register_process(process)
    try:
        _check_cancel(cancelled)
        _, text = process.communicate()
        _check_cancel(cancelled)
        if process.returncode:
            raise ValueError("Audio loudness analysis failed: {}".format(
                text[-2000:]
            ))
        matches = re.findall(r'\{\s*"input_i"\s*:.*?\}', text, re.DOTALL)
        if not matches:
            raise ValueError("FFmpeg loudnorm did not return audio measurements.")
        try:
            measurements = json.loads(matches[-1])
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid loudnorm JSON: {}".format(exc))
        if not math.isfinite(float(measurements["input_i"])):
            return None
        _loudnorm_filter(measurements)
        return measurements
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if process.stderr:
            process.stderr.close()
        if register_process is not None:
            register_process(None)


def apply_youtube_analysis(
    command, white_nits, light_levels=None, audio_measurements=None, grade=None
):
    """Replace x265 metadata after grading; NVENC has no x265-specific SEI path."""
    output = list(command)
    if light_levels is not None and "-x265-params" in output:
        max_cll, max_fall = light_levels[:2]
        position = output.index("-x265-params") + 1
        output[position] = x265_sdr_mastering_params(
            white_nits, max_cll, max_fall, grade=grade
        )
    if audio_measurements is not None:
        audio_position = output.index("-c:a")
        output[audio_position:audio_position] = [
            "-af", _loudnorm_filter(audio_measurements)
        ]
    return output
