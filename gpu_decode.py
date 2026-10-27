"""Optional NVDEC input decoding; never substitutes NVENC for x265."""
from dataclasses import dataclass
import json
import numpy as np
from gpu_colour import _probe_run
from hdr_processing import build_sdr_to_linear_filter


@dataclass(frozen=True)
class DecodePlan:
    software_format: str

    def apply(self, command):
        if self.software_format not in ('nv12', 'p010le'):
            raise ValueError('Unsupported CUDA download format.')
        result = list(command)
        index = result.index('-i')
        result[index:index] = ['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda']
        filter_index = result.index('-vf') + 1
        result[filter_index] = 'hwdownload,format={},'.format(self.software_format) + result[filter_index]
        return result


def choose_nvidia_decoder(ffmpeg, ffprobe, path, profile, log=None,
                           cancelled=None, register_process=None):
    """Try only known 4:2:0 formats, and compare a decoded frame with software.

    Capability is source-dependent; an installed CUDA driver alone proves nothing.
    Download to CPU is explicit because zscale remains our colour reference.
    """
    if not ffprobe:
        return None
    try:
        probe = _probe_run([ffprobe, '-v', 'error', '-select_streams', 'v:0',
                           '-show_entries', 'stream=codec_name,pix_fmt', '-of', 'json', str(path)],
                          None, cancelled, register_process, timeout=15)
        stream = json.loads(probe)['streams'][0]
        if stream.get('codec_name') not in ('h264', 'hevc', 'vp9', 'av1'):
            return None
        formats = {'yuv420p':'nv12', 'yuvj420p':'nv12', 'yuv420p10le':'p010le'}
        if stream.get('pix_fmt') not in formats:
            return None
        plan = DecodePlan(formats[stream['pix_fmt']])
        filt = build_sdr_to_linear_filter(profile).replace('zscale=', 'zscale=w=64:h=64:', 1)
        cmd = [ffmpeg, '-v', 'error', '-nostdin', '-i', str(path), '-map', '0:v:0',
               '-an', '-sn', '-dn', '-frames:v', '1', '-vf', filt,
               '-pix_fmt', 'gbrpf32le', '-f', 'rawvideo', 'pipe:1']
        accelerated = _probe_run(plan.apply(cmd), None, cancelled, register_process, timeout=15)
        reference = _probe_run(cmd, None, cancelled, register_process, timeout=15)
        if len(accelerated) != 64*64*12 or len(reference) != len(accelerated):
            raise ValueError('Incomplete hardware decoder probe.')
        if not np.allclose(np.frombuffer(accelerated,dtype='<f4'),
                           np.frombuffer(reference,dtype='<f4'),rtol=1e-4,atol=2e-4):
            raise ValueError('Hardware decoder colour samples differ from software reference.')
        if log:
            log('GPU input decoding: NVIDIA NVDEC; CPU zscale and selected encoder retained.')
        return plan
    except (ValueError, OSError, KeyError, IndexError, TypeError) as exc:
        if log:
            log('NVDEC unavailable for this source; software decoding retained. ' + str(exc)[-300:])
        return None
