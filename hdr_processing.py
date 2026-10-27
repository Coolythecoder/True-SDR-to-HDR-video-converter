"""Validated SDR colour profiles and display-referred BT.709 -> BT.2100 PQ mapping.

The mapping uses FFmpeg's zscale/zimg implementation. SDR values are
linearized before changing primaries. PQ encoding is performed only after the
display-light reference white is established; changing metadata alone is not
a colour transform.

MovieLabs "Best Practices for Mapping BT.709 Content to HDR10", v1.0:
SDR BT.1886/100 nits -> linear BT.2020 -> 200-nit reference white -> PQ.
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path


DEFAULT_SDR_PROFILE = {
    "matrix": "bt709",
    "primaries": "bt709",
    "transfer": "bt709",
    "range": "limited",
}

# These are supported by FFmpeg's zscale filter; values are not interpolated
# into a shell command without first checking these allow-lists.
MATRICES = {
    "bt709": "bt709",
    "smpte170m": "smpte170m",
    "bt470bg": "bt470bg",
    "smpte240m": "smpte240m",
    "bt2020nc": "bt2020nc",
}
PRIMARIES = {
    "bt709": "bt709",
    "smpte170m": "smpte170m",
    "bt470bg": "bt470bg",
    "smpte240m": "smpte240m",
    "bt2020": "bt2020",
}
TRANSFERS = {
    "bt709": "bt709",
    "smpte170m": "smpte170m",
    "bt470bg": "bt470bg",
    "smpte240m": "smpte240m",
    "bt2020-10": "bt2020-10",
    "bt2020-12": "bt2020-12",
    "iec61966-2-1": "iec61966-2-1",
}
HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}
UNSPECIFIED = {"", "unknown", "unspecified", None}


def find_ffprobe(ffmpeg_exe):
    """Prefer ffprobe shipped alongside the FFmpeg executable."""
    ffmpeg_file = Path(ffmpeg_exe)
    sibling_name = "ffprobe.exe" if ffmpeg_file.suffix.lower() == ".exe" else "ffprobe"
    sibling = ffmpeg_file.with_name(sibling_name)
    if sibling.is_file() and os.access(str(sibling), os.X_OK):
        return str(sibling)
    return shutil.which(sibling_name) or shutil.which("ffprobe")


def has_zscale(ffmpeg_exe):
    """Fail closed instead of silently producing mislabelled HDR."""
    try:
        result = subprocess.run(
            [ffmpeg_exe, "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and bool(
        re.search(r"(?m)^\s*[.A-Z]{3}\s+zscale\s", result.stdout)
    )


def probe_sdr_profile(ffprobe_exe, input_path):
    """Read actual input colour tags. Missing tags use the documented 709 default.

    Raises ValueError for known HDR sources or other declared formats that
    cannot be converted safely by our zscale recipe.
    """
    if not ffprobe_exe:
        return DEFAULT_SDR_PROFILE.copy()
    try:
        result = subprocess.run(
            [
                ffprobe_exe, "-v", "error", "-select_streams", "v:0",
                "-show_entries",
                "stream=color_space,color_primaries,color_transfer,color_range",
                "-of", "json", input_path,
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if result.returncode:
            raise ValueError("FFprobe failed for {}: {}".format(
                Path(input_path).name, result.stderr.strip() or "unknown error"
            ))
        streams = json.loads(result.stdout).get("streams", [])
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        raise ValueError("Could not inspect source colour metadata: {}".format(exc))
    if not streams:
        raise ValueError("No video stream found in {}.".format(Path(input_path).name))

    source = streams[0]
    transfer = source.get("color_transfer")
    if transfer in HDR_TRANSFERS:
        raise ValueError(
            "{} is already HDR ({}); SDR-to-HDR conversion is unsafe.".format(
                Path(input_path).name, transfer
            )
        )

    def checked(field, choices, default):
        raw = source.get(field)
        if raw in UNSPECIFIED:
            return default
        if raw not in choices:
            raise ValueError(
                "{} has unsupported {}='{}'; refusing an incorrect colour conversion.".format(
                    Path(input_path).name, field, raw
                )
            )
        return choices[raw]

    primaries = checked("color_primaries", PRIMARIES, "bt709")
    matrix = checked("color_space", MATRICES, "bt709")
    transfer = checked("color_transfer", TRANSFERS, "bt709")
    range_tag = source.get("color_range")
    if range_tag in ("pc", "jpeg"):
        pixel_range = "full"
    elif range_tag in ("tv", "mpeg") or range_tag in UNSPECIFIED:
        pixel_range = "limited"
    else:
        raise ValueError("{} has unsupported color_range='{}'.".format(
            Path(input_path).name, range_tag
        ))
    return {
        "matrix": matrix,
        "primaries": primaries,
        "transfer": transfer,
        "range": pixel_range,
    }


def build_sdr_to_pq_filter(profile, white_nits=200, pixel_format="yuv420p10le"):
    """Construct physically meaningful 10-bit PQ and BT.2020 pixel conversion.

    zscale handles the SDR transfer (BT.1886-like display light), converting
    to linear RGB in BT.2020. The second pass applies 200-nit white -> PQ;
    its npl parameter is the *content white level*, NOT a 1000-nit peak.
    """
    if not isinstance(white_nits, int) or not 80 <= white_nits <= 400:
        raise ValueError("HDR reference white must be between 80 and 400 nits.")
    if pixel_format not in ("yuv420p10le", "p010le"):
        raise ValueError("Unsupported HDR pixel format.")
    for key, choices in (
        ("matrix", MATRICES.values()),
        ("primaries", PRIMARIES.values()),
        ("transfer", TRANSFERS.values()),
        ("range", ("limited", "full")),
    ):
        if profile.get(key) not in choices:
            raise ValueError("Unsupported source {}".format(key))

    return (
        "zscale=rin={range}:min={matrix}:pin={primaries}:tin={transfer}:"
        "m=gbr:p=bt2020:t=linear:r=full,"
        "format=gbrpf32le,"
        "zscale=t=smpte2084:m=bt2020nc:p=bt2020:r=limited:"
        "npl={white_nits}:d=error_diffusion,"
        "format={pixel_format}"
    ).format(white_nits=white_nits, pixel_format=pixel_format, **profile)


def x265_sdr_mastering_params(white_nits=200, max_cll=None, max_fall=0):
    """Default static SEI based on the documented BT.709 SDR mastering source.

    BT.709 source display primaries, D65 and output white level per MovieLabs.
    Unknown MaxFALL is explicitly signalled as zero, not guessed.
    """
    if not isinstance(white_nits, int) or not 80 <= white_nits <= 400:
        raise ValueError("Invalid HDR white level.")
    if max_cll is None:
        max_cll = white_nits
    if not isinstance(max_cll, int) or not 1 <= max_cll <= 10000:
        raise ValueError("MaxCLL must be from 1 to 10000.")
    if not isinstance(max_fall, int) or not 0 <= max_fall <= max_cll:
        raise ValueError("MaxFALL must be zero (unknown) or <= MaxCLL.")

    master_display = (
        "G(15000,30000)B(7500,3000)R(32000,16500)"
        "WP(15635,16450)L({},0)".format(white_nits * 10000)
    )
    return "hdr10=1:repeat-headers=1:master-display={}:max-cll={},{}".format(
        master_display, max_cll, max_fall
    )
