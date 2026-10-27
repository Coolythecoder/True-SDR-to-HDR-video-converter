"""Optional display-linear BT.2020 grading, before PQ encoding.

All RGB components are processed in float32. These are creative adjustments,
not recovery of lost highlights or a limited/full video-range conversion.
The identical FFmpeg expressions are used by the content-light scan and encode.
"""
from dataclasses import dataclass
import json
import math
import subprocess


@dataclass(frozen=True)
class CreativeHDRSettings:
    enabled: bool = False
    peak_nits: float = 600.0
    strength: float = 0.5
    knee: float = 0.75
    rolloff: float = 0.3
    exposure: float = 0.0
    contrast: float = 1.0
    midtones: float = 0.0
    shadows: float = 0.0
    black_floor: float = 0.0
    saturation: float = 1.0
    vibrance: float = 0.0
    temperature: float = 0.0
    tint: float = 0.0

    def validate(self, white_nits=200):
        if type(self.enabled) is not bool:
            raise ValueError('Creative HDR enabled must be a boolean.')
        limits = {
            'peak_nits': (200, 4000), 'strength': (0, 1),
            'knee': (0.5, 0.95), 'rolloff': (0.05, 1),
            'exposure': (-2, 2), 'contrast': (0.7, 1.4),
            'midtones': (-1, 1), 'shadows': (-2, 2),
            'black_floor': (0, 2), 'saturation': (0, 1.5),
            'vibrance': (-0.5, 1), 'temperature': (-1, 1), 'tint': (-1, 1),
        }
        if not isinstance(white_nits, (int, float)) or not math.isfinite(white_nits) or not 80 <= white_nits <= 400:
            raise ValueError('HDR reference white must be between 80 and 400 nits.')
        for name, (low, high) in limits.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError('Invalid creative HDR {} ({} to {}).'.format(name, low, high))
        if self.enabled and self.peak_nits < white_nits:
            raise ValueError('HDR peak ceiling cannot be below reference white.')
        return self


def active_grade(grade):
    if grade is not None and not isinstance(grade, CreativeHDRSettings):
        raise ValueError('Expected CreativeHDRSettings.')
    return grade is not None and grade.enabled


def _n(value):
    return format(float(value), '.12g')


def _geq(expressions):
    # Every argument is generated from validated finite numbers, not user text.
    return 'geq=' + ':'.join("{}='{}'".format(c, e) for c, e in zip('rgb', expressions)) + ':interpolation=nearest'


def build_creative_filter(grade=None, white_nits=200):
    """Return a float RGB filter suffix, empty for the original accurate mode.

    Highlight expansion is a smooth quadratic above a luminance knee. The peak
    setting is a ceiling, not a guarantee that any pixel reaches that ceiling.
    An exponential shoulder approaches it continuously. A common chroma scale
    compresses colours into the target RGB cube without separate channel clips.
    """
    if not active_grade(grade):
        return ''
    grade.validate(white_nits)
    rgb = [c + '(X,Y)' for c in 'rgb']
    lum = '0.2627*r(X,Y)+0.678*g(X,Y)+0.0593*b(X,Y)'
    chain = []
    # Creative warm/cool and green/magenta gains, not a Kelvin calibration.
    gains = [2 ** (grade.temperature * .35 + grade.tint * .15),
             2 ** (-grade.tint * .3),
             2 ** (-grade.temperature * .35 + grade.tint * .15)]
    normalization = sum(w * g for w, g in zip((.2627, .678, .0593), gains))
    gains = [g / normalization * 2 ** grade.exposure for g in gains]
    if any(abs(g - 1) > 1e-12 for g in gains):
        chain.append(_geq(['max(0,{})*{}'.format(c, _n(g)) for c, g in zip(rgb, gains)]))

    if (grade.contrast != 1 or grade.midtones != 0 or grade.shadows != 0 or grade.black_floor != 0):
        # Luminance-only tone adjustments; common RGB scaling preserves ratios.
        program = 'st(0,max(0,{}));st(1,0.18*pow(ld(0)/0.18,{}));'.format(lum, _n(grade.contrast))
        program += 'st(2,ld(1)/(ld(1)+0.18));'
        program += 'st(1,ld(1)*pow(2,{}*4*ld(2)*(1-ld(2))));'.format(_n(grade.midtones))
        program += 'st(1,ld(1)*pow(2,{}*exp(-ld(1)/0.12)));'.format(_n(grade.shadows))
        program += 'st(1,ld(1)+{}*exp(-ld(1)/0.05));'.format(_n(grade.black_floor / white_nits))
        chain.append(_geq([program + 'if(gt(ld(0),1e-12),{}*ld(1)/ld(0),ld(1))'.format(c) for c in rgb]))

    if grade.saturation != 1 or grade.vibrance != 0:
        program = 'st(0,max(0,{}));'.format(lum)
        program += 'st(1,max(r(X,Y),max(g(X,Y),b(X,Y))));st(2,min(r(X,Y),min(g(X,Y),b(X,Y))));'
        program += 'st(3,{}*(1+{}*pow(1-clip((ld(1)-ld(2))/max(ld(1),1e-12),0,1),2)));'.format(_n(grade.saturation), _n(grade.vibrance))
        chain.append(_geq([program + 'ld(0)+({}-ld(0))*ld(3)'.format(c) for c in rgb]))

    cap = grade.peak_nits / white_nits
    boost = grade.strength * (cap - 1)
    shoulder = cap * (1 - 0.5 * grade.rolloff)
    span = cap - shoulder
    program = 'st(0,max(0,{}));'.format(lum)
    program += 'st(1,ld(0)+{}*pow(max((ld(0)-{})/{},0),2));'.format(_n(boost), _n(grade.knee), _n(1 - grade.knee))
    program += 'st(2,if(lte(ld(1),{}),ld(1),{}+{}*(1-exp(-(ld(1)-{})/{}))));'.format(_n(shoulder), _n(shoulder), _n(span), _n(shoulder), _n(span))
    program += 'st(3,ld(2)/max(ld(0),1e-12));'
    program += 'st(4,min(r(X,Y),min(g(X,Y),b(X,Y)))*ld(3));st(5,max(r(X,Y),max(g(X,Y),b(X,Y)))*ld(3));'
    program += 'st(6,min(1,min(ld(2)/max(ld(2)-ld(4),1e-12),({}-ld(2))/max(ld(5)-ld(2),1e-12))));'.format(_n(cap))
    chain.append(_geq([program + 'clip(ld(2)+ld(6)*({}*ld(3)-ld(2)),0,{})'.format(c, _n(cap)) for c in rgb]))
    return ',' + ','.join(chain) + ',format=gbrpf32le'


def creative_mastering_params(grade, white_nits, max_cll=None, max_fall=0):
    """Declare a virtual BT.2020 mastering target, not a measured monitor.

    Until the scan finishes, content-light values are unknown (0,0), rather
    than pretending that the selected peak is the measured content maximum.
    """
    grade.validate(white_nits)
    if max_cll is None:
        max_cll = 0
    if type(max_cll) is not int or not 0 <= max_cll <= 10000:
        raise ValueError('Invalid creative MaxCLL.')
    if type(max_fall) is not int or not 0 <= max_fall <= max_cll:
        raise ValueError('Invalid creative MaxFALL.')
    return ('hdr10=1:repeat-headers=1:master-display='
            'G(8500,39850)B(6550,2300)R(35400,14600)'
            'WP(15635,16450)L({},0):max-cll={},{}').format(
                round(grade.peak_nits * 10000), max_cll, max_fall)


def require_creative_filters(ffmpeg_exe):
    result = subprocess.run([ffmpeg_exe, '-hide_banner', '-filters'],
                            capture_output=True, text=True, timeout=15)
    names = {parts[1] for line in result.stdout.splitlines()
             for parts in [line.split()] if len(parts) >= 2}
    if result.returncode or not {'geq', 'zscale', 'format'}.issubset(names):
        raise ValueError('Creative HDR requires FFmpeg geq, zscale and format filters.')


def probe_grade_geometry(ffprobe_exe, path):
    result = subprocess.run([ffprobe_exe, '-v', 'error', '-select_streams', 'v:0',
                             '-show_entries', 'stream=width,height', '-of', 'json', str(path)],
                            capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise ValueError('Cannot inspect creative HDR source: ' + result.stderr[-500:])
    try:
        video = json.loads(result.stdout)['streams'][0]
        if video['width'] < 2 or video['height'] < 2:
            raise ValueError('Invalid dimensions.')
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise ValueError('Cannot read creative HDR frame dimensions.') from exc
    return video
