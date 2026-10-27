"""Integrate the same grade with standard and YouTube HDR command builders."""
from creative_hdr import active_grade
from hdr_processing import build_sdr_to_pq_filter, x265_sdr_mastering_params


def grade_command(command, profile, white_nits, grade=None):
    """Return a new command; disabled grading is a byte-for-byte argument no-op.

    Preserve audio, quality, stream mapping, timing, and container flags. The
    creative master initially signals unknown light levels, populated by the
    post-grade analysis pass before encoding. NVENC retains its documented
    lack of static mastering-display SEI in this application.
    """
    result = list(command)
    if not active_grade(grade):
        return result
    grade.validate(white_nits)
    try:
        pixel_format = result[result.index('-pix_fmt') + 1]
        encoder = result[result.index('-c:v') + 1]
        filter_index = result.index('-vf') + 1
    except (ValueError, IndexError) as exc:
        raise ValueError('Creative HDR requires an existing 10-bit HDR command.') from exc
    if encoder not in ('libx265', 'hevc_nvenc') or pixel_format not in ('yuv420p10le', 'p010le'):
        raise ValueError('Creative HDR is only available for 10-bit HEVC output.')
    result[filter_index] = build_sdr_to_pq_filter(
        profile, white_nits, pixel_format, grade=grade
    )
    if encoder == 'libx265':
        metadata = x265_sdr_mastering_params(white_nits, grade=grade)
        if '-x265-params' in result:
            result[result.index('-x265-params') + 1] = metadata
        else:
            result[-1:-1] = ['-x265-params', metadata]
    return result
