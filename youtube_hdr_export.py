"""High-quality YouTube HDR10 upload preset and post-encode verification.

This preset deliberately uses libx265 rather than an NVENC fast path because
libx265 embeds the HDR10 static metadata (ST 2086 and MaxCLL/MaxFALL) that
YouTube recommends for PQ. It reuses the application's proven SDR->PQ filter.
"""

import json
import struct
import subprocess
from fractions import Fraction
from pathlib import Path

from hdr_processing import build_sdr_to_pq_filter, x265_sdr_mastering_params

YOUTUBE_SUFFIX = "_YouTube_HDR"
YOUTUBE_CRF = 16
YOUTUBE_FPS = (Fraction(24000, 1001), Fraction(24), Fraction(25),
               Fraction(30000, 1001), Fraction(30), Fraction(48),
               Fraction(50), Fraction(60000, 1001), Fraction(60))


def supports_libx265(ffmpeg_exe):
    """Don't silently substitute an encoder incapable of writing HDR10 SEIs."""
    try:
        completed = subprocess.run(
            [ffmpeg_exe, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if completed.returncode:
        return False
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "libx265":
            return True
    return False


def _probe(ffprobe_exe, path, show_entries):
    try:
        result = subprocess.run(
            [ffprobe_exe, "-v", "error", "-show_entries", show_entries,
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode:
            raise ValueError(result.stderr.strip() or "unknown ffprobe failure")
        return json.loads(result.stdout)
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        raise ValueError("Unable to inspect video using FFprobe: {}".format(exc))


def _supported_fps(value):
    try:
        fps = Fraction(value)
    except (ValueError, TypeError, ZeroDivisionError):
        return False
    return fps > 0 and any(abs(float(fps - allowed)) < 0.05 for allowed in YOUTUBE_FPS)


def validate_youtube_source(ffprobe_exe, input_path, colour_profile):
    """Only claim YouTube HDR readiness for standard progressive BT.709 footage."""
    if colour_profile.get("primaries") != "bt709":
        raise ValueError("YouTube HDR preset currently requires BT.709 SDR colour primaries.")
    data = _probe(
        ffprobe_exe, input_path,
        "stream=index,codec_type,width,height,field_order,avg_frame_rate,r_frame_rate",
    )
    streams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    if not streams:
        raise ValueError("The source has no video stream.")
    video = streams[0]
    width, height = video.get("width", 0), video.get("height", 0)
    if width < 2 or height < 2 or width % 2 or height % 2:
        raise ValueError("YouTube HDR export requires even video dimensions.")
    if min(width, height) < 720:
        raise ValueError("YouTube HDR export requires at least 720 pixels on the short edge.")
    if video.get("field_order") in ("tt", "bb", "tb", "bt"):
        raise ValueError("Deinterlace this source before YouTube HDR export.")
    fps = video.get("avg_frame_rate")
    if not _supported_fps(fps):
        fps = video.get("r_frame_rate")
    if not _supported_fps(fps):
        raise ValueError(
            "Unsupported source frame rate for this YouTube HDR preset: {}.".format(fps)
        )
    return video


def build_youtube_hdr_command(ffmpeg_exe, input_path, output_path, colour_profile,
                              white_nits=200, crf=YOUTUBE_CRF,
                              max_cll=None, max_fall=0):
    """Keep original frame rate and size, and embed real static HDR10 metadata."""
    if Path(output_path).suffix.lower() != ".mp4":
        raise ValueError("YouTube HDR export requires an .mp4 destination.")
    if colour_profile.get("primaries") != "bt709":
        raise ValueError("YouTube HDR export requires BT.709 SDR input.")
    if type(crf) is not int or not 0 <= crf <= 28:
        raise ValueError("YouTube HDR CRF must be an integer between 0 and 28.")
    colour_filter = build_sdr_to_pq_filter(
        colour_profile, white_nits, pixel_format="yuv420p10le"
    )
    mastering = x265_sdr_mastering_params(white_nits, max_cll, max_fall)
    return [
        ffmpeg_exe, "-y", "-i", str(input_path),
        "-map", "0:v:0", "-map", "0:a:0?", "-sn", "-dn",
        "-vf", colour_filter,
        "-c:v", "libx265", "-profile:v", "main10", "-preset", "medium",
        "-crf", str(crf), "-pix_fmt", "yuv420p10le",
        "-color_primaries", "bt2020", "-color_trc", "smpte2084",
        "-colorspace", "bt2020nc", "-color_range", "tv",
        "-x265-params", mastering,
        "-tag:v", "hvc1",
        "-c:a", "aac", "-b:a", "384k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart+negative_cts_offsets",
        "-use_editlist", "0", str(output_path),
    ]


def _mp4_moov_before_mdat(path):
    """Inspect top-level MP4 boxes without reading video payload into memory."""
    with open(path, "rb") as f:
        size = Path(path).stat().st_size
        position = 0
        moov = None
        mdat = None
        for _ in range(128):
            if position + 8 > size:
                break
            f.seek(position)
            hdr = f.read(8)
            box_size, box_type = struct.unpack(">I4s", hdr)
            header_size = 8
            if box_size == 1:
                if position + 16 > size:
                    break
                box_size = struct.unpack(">Q", f.read(8))[0]
                header_size = 16
            elif box_size == 0:
                box_size = size - position
            if box_size < header_size or position + box_size > size:
                raise ValueError("Corrupt MP4 box structure.")
            if box_type == b"moov" and moov is None:
                moov = position
            if box_type == b"mdat" and mdat is None:
                mdat = position
            position += box_size
            if moov is not None and mdat is not None:
                break
        return moov is not None and mdat is not None and moov < mdat


def _mp4_boxes(file, first, last):
    """Walk MP4 box headers safely without loading potentially huge media data."""
    position = first
    for _ in range(500):
        if position + 8 > last:
            break
        file.seek(position)
        box_size, box_type = struct.unpack(">I4s", file.read(8))
        header = 8
        if box_size == 1:
            if position + 16 > last:
                raise ValueError("Truncated MP4 large-size box.")
            box_size = struct.unpack(">Q", file.read(8))[0]
            header = 16
        elif box_size == 0:
            box_size = last - position
        if box_size < header or position + box_size > last:
            raise ValueError("Invalid MP4 box size.")
        yield box_type, position + header, position + box_size
        position += box_size


def _mp4_has_edit_list(path):
    """YouTube recommends MP4 uploads without moov/trak/edts edit lists."""
    with open(path, "rb") as file:
        size = Path(path).stat().st_size
        for box, start, end in _mp4_boxes(file, 0, size):
            if box == b"moov":
                for child, child_start, child_end in _mp4_boxes(file, start, end):
                    if child == b"trak":
                        for track_child, _, _ in _mp4_boxes(
                            file, child_start, child_end
                        ):
                            if track_child == b"edts":
                                return True
                return False
    raise ValueError("MP4 moov atom is missing.")


def verify_youtube_hdr_file(ffprobe_exe, path):
    """Prove the encoded MP4 matches the core YouTube HDR requirements.

    Does not imply YouTube has uploaded/processed it or guarantee a specific
    bitrate, mastering aesthetics or playback device behaviour.
    """
    if not Path(path).is_file() or Path(path).stat().st_size < 100:
        raise ValueError("YouTube HDR output was not created.")
    data = _probe(
        ffprobe_exe, path,
        "stream=index,codec_type,codec_name,profile,pix_fmt,color_primaries,"
        "color_transfer,color_space,color_range,width,height,avg_frame_rate,sample_rate,channels:format=format_name",
    )
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError("YouTube HDR validation: no video stream found.")
    expected = {
        "codec_name": "hevc",
        "profile": "Main 10",
        "pix_fmt": "yuv420p10le",
        "color_primaries": "bt2020",
        "color_transfer": "smpte2084",
        "color_space": "bt2020nc",
        "color_range": "tv",
    }
    for key, value in expected.items():
        if video.get(key) != value:
            raise ValueError(
                "YouTube HDR validation failed: {} was {!r}, expected {!r}.".format(
                    key, video.get(key), value
                )
            )
    if not _supported_fps(video.get("avg_frame_rate")):
        raise ValueError("YouTube HDR validation: unsupported output frame rate.")
    if min(video.get("width", 0), video.get("height", 0)) < 720:
        raise ValueError("YouTube HDR validation: output resolution too low.")
    audio = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
    if audio:
        if audio[0].get("codec_name") != "aac" or audio[0].get("sample_rate") != "48000" or audio[0].get("channels") != 2:
            raise ValueError("YouTube HDR validation: audio is not stereo AAC at 48 kHz.")
    if not _mp4_moov_before_mdat(path):
        raise ValueError("YouTube HDR validation: MP4 fast-start moov atom missing.")
    if _mp4_has_edit_list(path):
        raise ValueError("YouTube HDR validation: MP4 contains edit lists (edts).")

    try:
        output = subprocess.run(
            [ffprobe_exe, "-v", "error", "-select_streams", "v:0",
             "-read_intervals", "%+#3", "-show_frames", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        if output.returncode:
            raise ValueError(output.stderr.strip() or "FFprobe could not inspect SEI")
        frames = json.loads(output.stdout).get("frames", [])
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        raise ValueError("YouTube HDR validation: could not inspect SEI: {}".format(exc))
    observed = {
        item.get("side_data_type")
        for frame in frames for item in frame.get("side_data_list", [])
    }
    required = {"Mastering display metadata", "Content light level metadata"}
    if not required.issubset(observed):
        raise ValueError(
            "YouTube HDR validation: missing HEVC HDR10 SEI metadata: {}.".format(
                ", ".join(sorted(required - observed))
            )
        )
    return True
