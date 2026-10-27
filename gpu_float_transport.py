"""Lossless float-plane reassembly, including legacy FFmpeg mergeplanes.

Some FFmpeg builds copy only two bytes per sample in mergeplanes even when
its negotiated format is 32-bit float. The right half of each output row is
then uninitialised. OpenCL can have computed every pixel correctly and still
produce a half-black preview/export after this CPU-side copy.

Duplicate each grayscale row before merging, then retain its original left
half. Both the legacy two-byte copy and the corrected four-byte copy include
all of the original row. No arithmetic, scaling or integer conversion occurs.
"""
import re


LINEAR_BT2020_TAGS = (
    'setparams=range=full:color_primaries=bt2020:'
    'color_trc=linear:colorspace=gbr'
)


def merge_float_planes(prefix):
    """Consume [<prefix>or/og/ob] float planes and return one float GBR stream.

    hstack copies rows using the pixel format's actual byte stride. After
    mergeplanes the original row occupies the left half of the double-width
    image on both old and new FFmpeg. The explicit x=0 crop is essential: a
    default centred crop would include the legacy unwritten region.

    This changes only temporary CPU buffers, not the GPU input geometry,
    exported dimensions, sample aspect ratio, pixel values or timestamps.
    """
    if not isinstance(prefix, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', prefix):
        raise ValueError('Invalid internal float-merge label.')
    graph = []
    for channel in 'rgb':
        label = prefix + 'merge' + channel
        graph.append('[{0}o{1}]split=2[{2}a][{2}b]'.format(prefix, channel, label))
        graph.append('[{0}a][{0}b]hstack=inputs=2[{0}]'.format(label))
    graph.append(
        '[{0}mergeg][{0}mergeb][{0}merger]'
        'mergeplanes=0x001020:format=gbrpf32le,'
        'crop=w=iw/2:h=ih:x=0:y=0:exact=1,'.format(prefix) + LINEAR_BT2020_TAGS
    )
    return ';'.join(graph)
