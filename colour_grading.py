"""Opt-in display-linear Rec.2020 grading and bounded highlight expansion.

This is a creative inverse tone curve, not recovery of clipped source detail.
The FFmpeg and NumPy implementations share equations; production encoding and
content-light analysis both use the exact same float geq filter.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GradeSettings:
    exposure: float = 0.0       # stops
    contrast: float = 1.0       # luminance power, pivot 0.18
    saturation: float = 1.0
    warmth: float = 0.0         # creative red/blue balance, not Kelvin
    expand: bool = False
    strength: float = 0.5
    knee: float = 0.75          # fraction of pre-expansion reference white
    peak_nits: int = 1000       # ceiling, NOT an assertion of measured MaxCLL

    def validate(self, white_nits=200):
        bounds = {"exposure": (-2, 2), "contrast": (0.75, 1.25),
                  "saturation": (0, 1.5), "warmth": (-0.5, 0.5),
                  "strength": (0, 1), "knee": (0.5, 0.95)}
        for name, (low, high) in bounds.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError("Invalid grading {} ({} to {}).".format(name, low, high))
        if type(self.expand) is not bool:
            raise ValueError("Highlight expansion must be a boolean.")
        if type(self.peak_nits) is not int or not 400 <= self.peak_nits <= 4000:
            raise ValueError("Creative HDR peak must be an integer from 400 to 4000 nits.")
        if type(white_nits) is not int or not 80 <= white_nits <= 400:
            raise ValueError("Reference white must be 80 to 400 nits.")
        if self.expand and self.strength > 0 and self.peak_nits <= white_nits:
            raise ValueError("HDR highlight peak must exceed reference white.")
        return self

    @property
    def active(self):
        return (self.exposure != 0 or self.contrast != 1 or
                self.saturation != 1 or self.warmth != 0 or
                (self.expand and self.strength > 0))


def settings_or_default(settings=None, white_nits=200):
    if settings is None:
        settings = GradeSettings()
    if not isinstance(settings, GradeSettings):
        raise ValueError("Expected immutable GradeSettings.")
    return settings.validate(white_nits)


def creative_peak(settings=None, white_nits=200):
    """Return the synthetic working-volume ceiling, not measured light levels."""
    grade = settings_or_default(settings, white_nits)
    return grade.peak_nits if grade.active else white_nits


def _gains(grade):
    red, green, blue = 2.0 ** grade.warmth, 1.0, 2.0 ** -grade.warmth
    normalizer = 0.2627 * red + 0.6780 * green + 0.0593 * blue
    scale = 2.0 ** grade.exposure / normalizer
    return red * scale, green * scale, blue * scale


def build_grade_filter(settings=None, white_nits=200):
    """Return an optional float RGB geq suffix. Neutral settings bypass exactly.

    Every component uses the same luminance/magnitude gain. Saturation is
    reduced at the nonnegative RGB boundary instead of clipping individual
    channels. A shared exponential shoulder bounds the maximum component.
    FFmpeg st/ld state is local to each component/slice and reset per pixel.
    """
    grade = settings_or_default(settings, white_nits)
    if not grade.active:
        return ""
    red, green, blue = _gains(grade)
    expr = ["st(0,max(0,r(X,Y))*{:.12g})".format(red),
            "st(1,max(0,g(X,Y))*{:.12g})".format(green),
            "st(2,max(0,b(X,Y))*{:.12g})".format(blue)]
    if grade.contrast != 1 or grade.saturation != 1:
        expr += ["st(3,0.2627*ld(0)+0.6780*ld(1)+0.0593*ld(2))",
                 "st(4,pow(max(ld(3),1e-12)/0.18,{:.12g}))".format(grade.contrast - 1),
                 "st(5,min({0:.12g},ld(3)/max(ld(3)-min(ld(0),min(ld(1),ld(2))),1e-12)))".format(grade.saturation)]
        for channel in range(3):
            expr.append("st({0},max(0,(ld(3)+ld(5)*(ld({0})-ld(3)))*ld(4)))".format(channel))
    expr.append("st(6,max(ld(0),max(ld(1),ld(2))))")
    peak = grade.peak_nits / float(white_nits)
    if grade.expand and grade.strength > 0:
        expr.append("st(7,ld(6)+{:.12g}*pow(max(0,(ld(6)-{:.12g})/{:.12g}),2))".format(
            grade.strength * (peak - 1), grade.knee, 1 - grade.knee))
    else:
        expr.append("st(7,ld(6))")
    shoulder, headroom = 0.9 * peak, 0.1 * peak
    # max(0,...) in exp avoids overflow even in the unused branch.
    expr.append("st(8,if(lte(ld(7),{0:.12g}),ld(7),{0:.12g}+{1:.12g}*(1-exp(-max(0,ld(7)-{0:.12g})/{1:.12g}))))".format(shoulder, headroom))
    prefix = ";".join(expr) + ";"
    channels = []
    for name, channel in (("r", 0), ("g", 1), ("b", 2)):
        channels.append("{}='{}ld({})*ld(8)/max(ld(6),1e-12)'".format(name, prefix, channel))
    # No pixel neighbours or temporal state: nearest avoids interpolation work.
    return ",geq=" + ":".join(channels) + ":interpolation=nearest,format=gbrpf32le"


def grade_rgb(rgb, settings=None, white_nits=200):
    """NumPy reference for tests; last axis is R,G,B, values relative to white."""
    import numpy as np
    grade = settings_or_default(settings, white_nits)
    source = np.asarray(rgb, dtype=np.float64)
    if source.shape[-1:] != (3,) or not np.isfinite(source).all():
        raise ValueError("Expected finite RGB values with last dimension 3.")
    if not grade.active:
        return source.copy()
    result = np.maximum(source, 0) * np.asarray(_gains(grade))
    if grade.contrast != 1 or grade.saturation != 1:
        luminance = np.sum(result * [0.2627, 0.6780, 0.0593], axis=-1, keepdims=True)
        gain = np.power(np.maximum(luminance, 1e-12) / 0.18, grade.contrast - 1)
        sat = np.minimum(grade.saturation, luminance / np.maximum(luminance - result.min(axis=-1, keepdims=True), 1e-12))
        result = np.maximum(0, luminance + sat * (result - luminance)) * gain
    maximum = result.max(axis=-1, keepdims=True)
    peak = grade.peak_nits / float(white_nits)
    expanded = maximum.copy()
    if grade.expand and grade.strength > 0:
        expanded += grade.strength * (peak - 1) * np.maximum(0, (maximum - grade.knee) / (1 - grade.knee)) ** 2
    shoulder, headroom = 0.9 * peak, 0.1 * peak
    mapped = np.where(expanded <= shoulder, expanded,
                      shoulder + headroom * (1 - np.exp(-np.maximum(0, expanded - shoulder) / headroom)))
    return result * mapped / np.maximum(maximum, 1e-12)
