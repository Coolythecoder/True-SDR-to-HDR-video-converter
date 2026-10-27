"""Read-only, reduced-resolution live grading preview recipes.

The export's display-linear BT.2020 grading expressions are reused exactly.
Only the viewing transform is different: an SDR/sRGB approximation with a
fixed 1000-nit scale, NOT a native HDR surface or a mastering monitor.
"""
from dataclasses import dataclass
from fractions import Fraction
import math
from pathlib import Path
from creative_hdr import CreativeHDRSettings, build_creative_filter
from hdr_processing import (
    DEFAULT_SDR_PROFILE, MATRICES, PRIMARIES, TRANSFERS, HDR_TRANSFERS,
    build_sdr_to_linear_filter,
)

PREVIEW_WIDTHS = (480, 640, 960)
VIEW_PEAK_NITS = 1000


@dataclass(frozen=True)
class PreviewSource:
    path: str
    width: int
    height: int
    duration: float
    fps: float
    profile: dict
    display_aspect: float
    assumptions: tuple = ()

    def proxy_size(self, bound=640):
        if bound not in PREVIEW_WIDTHS:
            raise ValueError('Unsupported preview resolution.')
        aspect = min(20.0, max(0.05, self.display_aspect))
        width = min(bound, bound * 9 / 16 * aspect)
        height = width / aspect
        return max(2, round(width / 2) * 2), max(2, round(height / 2) * 2)


def _number(value, default=0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (ValueError, TypeError):
        return default


def source_from_probe(path, data):
    """Validate FFprobe data without silently accepting HDR or unsupported tags."""
    videos = [s for s in data.get('streams', []) if s.get('codec_type') == 'video']
    if not videos:
        raise ValueError('No video stream found.')
    stream = videos[0]
    width, height = int(stream.get('width', 0)), int(stream.get('height', 0))
    if not (2 <= width <= 32768 and 2 <= height <= 32768):
        raise ValueError('Unsupported source dimensions.')
    if stream.get('color_transfer') in HDR_TRANSFERS:
        raise ValueError('This is already HDR. Preview the original SDR source instead.')
    assumptions = []
    profile = {}
    for key, tag, choices in (
        ('matrix', 'color_space', MATRICES),
        ('primaries', 'color_primaries', PRIMARIES),
        ('transfer', 'color_transfer', TRANSFERS),
    ):
        raw = stream.get(tag)
        if raw in (None, '', 'unknown', 'unspecified'):
            profile[key] = DEFAULT_SDR_PROFILE[key]
            assumptions.append(tag)
        elif raw in choices:
            profile[key] = choices[raw]
        else:
            raise ValueError('Unsupported source {}: {}'.format(tag, raw))
    pixel_range = stream.get('color_range')
    if pixel_range in ('pc', 'jpeg'):
        profile['range'] = 'full'
    elif pixel_range in ('tv', 'mpeg'):
        profile['range'] = 'limited'
    elif pixel_range in (None, '', 'unknown', 'unspecified'):
        profile['range'] = 'limited'
        assumptions.append('color_range')
    else:
        raise ValueError('Unsupported source colour range.')
    duration = _number(stream.get('duration')) or _number(data.get('format', {}).get('duration'))
    if duration <= 0:
        raise ValueError('Cannot determine video duration for the preview timeline.')
    try:
        fps = float(Fraction(stream.get('avg_frame_rate', '0/1')))
    except (ValueError, TypeError, ZeroDivisionError):
        fps = 0
    if not math.isfinite(fps) or not 0 < fps <= 1000:
        fps = 30.0
    try:
        sar = float(Fraction(str(stream.get('sample_aspect_ratio', '1:1')).replace(':', '/')))
    except (ValueError, ZeroDivisionError):
        sar = 1.0
    if not math.isfinite(sar) or sar <= 0:
        sar = 1.0
    aspect = width * sar / height
    rotation = _number(stream.get('tags', {}).get('rotate'))
    for side in stream.get('side_data_list', []):
        if 'rotation' in side:
            rotation = _number(side['rotation'])
    if round(rotation / 90) % 2:
        aspect = 1.0 / aspect
    return PreviewSource(str(Path(path).resolve()), width, height, duration, fps,
                         profile, aspect, tuple(assumptions))


def probe_command(ffprobe, path):
    return [ffprobe, '-v', 'error', '-select_streams', 'v:0',
            '-show_streams', '-show_format', '-of', 'json', str(Path(path).resolve())]


def decode_command(ffmpeg, source, seconds, bound=640):
    """Decode one frame only; seeking never changes the exported source."""
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Invalid preview time.')
    width, height = source.proxy_size(bound)
    position = min(seconds, max(0, source.duration - 1 / source.fps))
    filt = build_sdr_to_linear_filter(source.profile).replace(
        'zscale=', 'zscale=w={}:h={}:'.format(width, height), 1)
    return [ffmpeg, '-hide_banner', '-nostdin', '-v', 'error',
            '-ss', '{:.6f}'.format(position), '-i', source.path,
            '-map', '0:v:0', '-an', '-sn', '-dn', '-frames:v', '1',
            '-vf', filt + ',setsar=1', '-pix_fmt', 'gbrpf32le',
            '-f', 'rawvideo', 'pipe:1']


def viewing_filter(white_nits=200):
    """Fixed SDR viewing scale for BOTH halves, independent of grade peak."""
    if not isinstance(white_nits, (int, float)) or not math.isfinite(white_nits) or not 80 <= white_nits <= 400:
        raise ValueError('HDR reference white must be between 80 and 400 nits.')
    gain = white_nits / 200.0
    gain_filter = ''
    if gain != 1:
        gain_filter = "geq=r='r(X,Y)*{0}':g='g(X,Y)*{0}':b='b(X,Y)*{0}',".format(gain)
    return (gain_filter + 'tonemap=mobius:param=0.3:desat=0:peak=5,'
            'zscale=pin=bt2020:tin=linear:min=gbr:rin=full:'
            'p=bt709:t=iec61966-2-1:m=gbr:r=full,'
            'format=gbrp,format=rgb24')


def render_command(ffmpeg, width, height, grade=None, white_nits=200):
    """Return before/after pairs from the SAME cached float source frame."""
    grade = grade or CreativeHDRSettings()
    grade.validate(white_nits)
    if not (2 <= width <= 960 and 2 <= height <= 540):
        raise ValueError('Invalid preview dimensions.')
    view = viewing_filter(white_nits)
    change = build_creative_filter(grade, white_nits).lstrip(',')
    after = (change + ',' if change else '') + view
    # Rawvideo transport does not carry metadata. Restore the known float
    # colour interpretation before the display transform, rather than guess.
    graph = ('[0:v]setparams=range=full:color_primaries=bt2020:'
             'color_trc=linear:colorspace=gbr,split=2[a][b];[a]' + view + '[before];[b]' + after +
             '[after];[before][after]hstack=inputs=2,format=rgb24[out]')
    return [ffmpeg, '-hide_banner', '-nostdin', '-v', 'error',
            '-f', 'rawvideo', '-pixel_format', 'gbrpf32le',
            '-video_size', '{}x{}'.format(width, height), '-framerate', '1',
            '-i', 'pipe:0', '-filter_complex_threads', '2', '-filter_complex', graph,
            '-map', '[out]', '-frames:v', '1', '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', 'pipe:1']
