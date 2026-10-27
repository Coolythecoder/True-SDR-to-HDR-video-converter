"""Input/output planning and FFmpeg discovery for the SDR-to-HDR GUI."""

import os
import shutil
import sys
from pathlib import Path

VIDEO_EXTENSIONS = frozenset((".mp4", ".mov", ".mkv"))


def find_ffmpeg():
    """Find FFmpeg, including Homebrew paths absent from macOS GUI PATH."""
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    if sys.platform == "darwin":
        for candidate in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/opt/local/bin/ffmpeg"):
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
    return None


def plan_conversion_jobs(input_path, output_path, batch_mode, output_suffix="_HDR"):
    """Return [(video_file, mp4_file), ...], never a directory as an input file.

    Normally the output is an existing folder. A manually entered .mp4 file
    path is accepted for a single conversion to preserve previous usage.
    """
    if output_suffix not in ("_HDR", "_SDR", "_YouTube_HDR"):
        raise ValueError("Unsupported output filename suffix.")

    source = Path(input_path).expanduser()
    destination = Path(output_path).expanduser()

    explicit_file = (
        not batch_mode
        and destination.suffix.lower() == ".mp4"
        and not destination.is_dir()
        and destination.parent.is_dir()
    )
    if not explicit_file and not destination.is_dir():
        raise ValueError("Select an existing output folder.")

    if batch_mode:
        if not source.is_dir():
            raise ValueError("Batch mode requires an existing input folder.")
        videos = sorted(
            (path for path in source.iterdir()
             if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS),
            key=lambda path: path.name.casefold(),
        )
        # Avoid re-processing last run's output when both folders are the same.
        if source.resolve() == destination.resolve():
            videos = [path for path in videos if not path.stem.casefold().endswith(("_hdr", "_sdr", "_youtube_hdr"))]
    else:
        if not source.is_file() or source.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError("Select an existing MP4, MOV or MKV input video.")
        videos = [source]

    if not videos:
        raise ValueError("No supported videos were found in the input folder.")

    jobs = []
    used_outputs = set()
    for video in videos:
        output = destination if explicit_file else destination / (video.stem + output_suffix + ".mp4")
        if output.resolve() == video.resolve():
            raise ValueError("Output file cannot overwrite the input video.")

        # e.g. clip.mov and clip.mkv must not both overwrite clip_HDR.mp4.
        if str(output.resolve()).casefold() in used_outputs:
            stem = "{}_{}{}".format(video.stem, video.suffix[1:].lower(), output_suffix)
            output = destination / (stem + ".mp4")
            number = 2
            while str(output.resolve()).casefold() in used_outputs:
                output = destination / ("{}_{}.mp4".format(stem, number))
                number += 1
        used_outputs.add(str(output.resolve()).casefold())
        jobs.append((str(video), str(output)))

    return jobs
