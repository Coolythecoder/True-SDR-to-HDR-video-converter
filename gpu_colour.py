"""Optional float32 OpenCL grade and full-pixel content-light reductions.

CPU zscale remains the colour-management reference. Only the creative equations
and reductions move to OpenCL; encoder/audio/timing choices are not replaced.
No CUDA toolkit or extra Python dependency is required. FFmpeg needs OpenCL.
"""
from dataclasses import dataclass
import math
from pathlib import Path
import re
import subprocess
import tempfile
import time

import numpy as np
from creative_hdr import CreativeHDRSettings, active_grade, build_creative_filter
from hdr_processing import build_sdr_to_linear_filter
from gpu_float_transport import merge_float_planes

# Explicitly require GPU devices in production. A numeric device is accepted by
# the test harness to exercise the kernels using a software OpenCL implementation.
GPU_DEVICES = (',device_type=gpu,device_vendor=NVIDIA',
               ',device_type=gpu,device_vendor=AMD', ',device_type=gpu')
TILE = 16


def _f(value):
    if not math.isfinite(value):
        raise ValueError('Non-finite GPU constant.')
    text = format(float(value), '.12g')
    return text + ('f' if '.' in text or 'e' in text else '.0f')


def kernel_source(grade=None, white_nits=200):
    """Port the existing grade without changing its curve or control meanings."""
    grade = grade or CreativeHDRSettings()
    grade.validate(white_nits)
    body = 'return rgb;'
    if active_grade(grade):
        gains = [2 ** (grade.temperature * .35 + grade.tint * .15),
                 2 ** (-grade.tint * .3),
                 2 ** (-grade.temperature * .35 + grade.tint * .15)]
        norm = sum(w * g for w, g in zip((.2627, .678, .0593), gains))
        gains = [g / norm * 2 ** grade.exposure for g in gains]
        lines = []
        if any(abs(g - 1) > 1e-12 for g in gains):
            lines.append('rgb = fmax(rgb, 0.0f) * (float3)(%s);' % ','.join(map(_f, gains)))
        if grade.contrast != 1 or grade.midtones or grade.shadows or grade.black_floor:
            lines += [
                'float y = fmax(luma(rgb), 0.0f);',
                'float t = 0.18f * pow(y/0.18f, %s);' % _f(grade.contrast),
                'float m = t / (t + 0.18f);',
                't *= pow(2.0f, %s*4.0f*m*(1.0f-m));' % _f(grade.midtones),
                't *= pow(2.0f, %s*exp(-t/0.12f));' % _f(grade.shadows),
                't += %s*exp(-t/0.05f);' % _f(grade.black_floor / white_nits),
                'rgb = y > 1e-12f ? rgb*(t/y) : (float3)(t);']
        if grade.saturation != 1 or grade.vibrance:
            lines += [
                'float sy = fmax(luma(rgb), 0.0f);',
                'float hi = fmax(rgb.x, fmax(rgb.y, rgb.z));',
                'float lo = fmin(rgb.x, fmin(rgb.y,rgb.z));',
                'float v = 1.0f-clamp((hi-lo)/fmax(hi,1e-12f),0.0f,1.0f);',
                'rgb = sy+(rgb-sy)*(%s*(1.0f+%s*v*v));' % (_f(grade.saturation), _f(grade.vibrance))]
        cap = grade.peak_nits / white_nits
        shoulder = cap * (1 - .5 * grade.rolloff)
        lines += [
            'float iy = fmax(luma(rgb), 0.0f);',
            'float h = fmax((iy-%s)/%s, 0.0f);' % (_f(grade.knee), _f(1-grade.knee)),
            'float boosted = iy+%s*h*h;' % _f(grade.strength*(cap-1)),
            'float oy = boosted <= %s ? boosted : %s+%s*(1.0f-exp(-(boosted-%s)/%s));' %
            (_f(shoulder), _f(shoulder), _f(cap-shoulder), _f(shoulder), _f(cap-shoulder)),
            'rgb *= oy/fmax(iy,1e-12f);',
            'float mn=fmin(rgb.x,fmin(rgb.y,rgb.z)), mx=fmax(rgb.x,fmax(rgb.y,rgb.z));',
            'float chroma=fmin(1.0f,fmin(oy/fmax(oy-mn,1e-12f),(%s-oy)/fmax(mx-oy,1e-12f)));' % _f(cap),
            'return clamp(oy+chroma*(rgb-oy),0.0f,%s);' % _f(cap)]
        body = '\n'.join(lines)
    code = '''// Generated from validated numeric controls. Not user-supplied code.
#pragma OPENCL FP_CONTRACT OFF
const sampler_t smp=CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP_TO_EDGE|CLK_FILTER_NEAREST;
float luma(float3 v) { return 0.2627f*v.x+0.6780f*v.y+0.0593f*v.z; }
float3 grade_pixel(float3 rgb) {
BODY
}
float3 pixel(read_only image2d_t r, read_only image2d_t g, read_only image2d_t b, int2 p) {
 return grade_pixel((float3)(read_imagef(r,smp,p).x,read_imagef(g,smp,p).x,read_imagef(b,smp,p).x));
}
'''.replace('BODY', body)
    for channel, component in zip('rgb', 'xyz'):
        code += '''kernel void grade_CHANNEL(write_only image2d_t dst, uint frame,
 read_only image2d_t r, read_only image2d_t g, read_only image2d_t b) {
 int2 p=(int2)(get_global_id(0),get_global_id(1));
 write_imagef(dst,p,(float4)(pixel(r,g,b,p).COMPONENT,0.0f,0.0f,1.0f));
}
'''.replace('CHANNEL', channel).replace('COMPONENT', component)
    code += '''kernel void light_stats(write_only image2d_t dst, uint frame,
 read_only image2d_t r, read_only image2d_t g, read_only image2d_t b) {
 int2 out=(int2)(get_global_id(0),get_global_id(1));
 int tiles_h=(get_image_height(r)+15)/16;
 int ty=out.y%tiles_h, sum_mode=out.y>=tiles_h;
 float peak=0.0f, sum=0.0f;
 for(int dy=0;dy<16;++dy) for(int dx=0;dx<16;++dx) {
   int2 p=(int2)(out.x*16+dx,ty*16+dy);
   if(p.x>=get_image_width(r)||p.y>=get_image_height(r)) continue;
   float3 c=pixel(r,g,b,p);
   if(!all(isfinite(c))) { peak=NAN; sum=NAN; break; }
   float y=clamp(luma(c),0.0f,MAX_LIGHT);
   peak=fmax(peak,y); sum+=y;
 }
 write_imagef(dst,out,(float4)(sum_mode?sum:peak,0.0f,0.0f,1.0f));
}
'''.replace('MAX_LIGHT', _f(10000 / white_nits))
    return code


def _path_option(path):
    # Two filter-parser layers. Outer quotes protect spaces, commas and brackets;
    # inner escaping protects the colon/quote consumed by the option parser.
    value = str(path).replace('\\', '/')
    value = value.replace("'", "'\\\\\\''").replace(':', '\\:')
    return "'" + value + "'"


class GPUPlan:
    """Owned shader lifetime. Close only after all commands using it have ended."""
    def __init__(self, grade=None, white_nits=200, device=GPU_DEVICES[0]):
        if device not in GPU_DEVICES and not re.fullmatch(r'\d+\.\d+', device):
            raise ValueError('Invalid OpenCL device selector.')
        self.grade = grade or CreativeHDRSettings()
        self.grade.validate(white_nits)
        self.white_nits, self.device = white_nits, device
        self._temp = tempfile.TemporaryDirectory(prefix='hdr_opencl_')
        self.shader = Path(self._temp.name) / 'grade.cl'
        self.shader.write_text(kernel_source(self.grade, white_nits), encoding='utf-8')

    def close(self):
        self._temp.cleanup()

    def device_args(self):
        return ['-init_hw_device', 'opencl=hdrcl:' + self.device,
                '-filter_hw_device', 'hdrcl']

    def wrap(self, command):
        return [command[0]] + self.device_args() + list(command[1:])

    def filter(self, stats_size=None, prefix='hgpu'):
        """Float GBR -> separate GPU channels -> grade -> float GBR / tile stats.

        program_opencl exposes the current plane, not all planes. Splitting into
        three float grayscale inputs gives every kernel access to all RGB values.
        No integer RGB intermediate or precision-losing packed conversion is used.
        """
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', prefix):
            raise ValueError('Invalid internal filter label.')
        branches = 1 if stats_size else 3
        graph = ['split=3[{0}r][{0}g][{0}b]'.format(prefix)]
        for c in 'rgb':
            label = '[{}{}]extractplanes={},format=grayf32le,hwupload'.format(prefix,c,c)
            if branches == 3:
                label += ',split=3' + ''.join('[{}{}{}]'.format(prefix,c,i) for i in range(3))
            else:
                label += '[{}{}0]'.format(prefix,c)
            graph.append(label)
        program = 'program_opencl=source={}:inputs=3:kernel='.format(_path_option(self.shader))
        if stats_size:
            width, height = stats_size
            tw, th = (width+15)//16, (height+15)//16
            graph.append(''.join('[{}{}0]'.format(prefix,c) for c in 'rgb') +
                         program + 'light_stats:size={}x{},hwdownload,format=grayf32le'.format(tw,th*2))
        else:
            for i, c in enumerate('rgb'):
                graph.append(''.join('[{}{}{}]'.format(prefix,s,i) for s in 'rgb') +
                             program + 'grade_{},hwdownload,format=grayf32le[{}o{}]'.format(c,prefix,c))
            # Legacy mergeplanes copies only half a float32 row. Reassemble
            # through a lossless double-width transport and left crop instead;
            # the same helper also restores the known linear BT.2020 tags.
            graph.append(merge_float_planes(prefix))
        return ';'.join(graph)

    def export_command(self, command, profile):
        """Replace ONLY the grade; encoder, CRF, audio, metadata and mux stay intact."""
        out = list(command)
        if not active_grade(self.grade):
            return out
        pos = out.index('-vf') + 1
        original = build_sdr_to_linear_filter(profile, grade=self.grade, white_nits=self.white_nits)
        if not out[pos].startswith(original + ',zscale='):
            raise ValueError('GPU grade cannot replace an unrecognized export filter.')
        base = build_sdr_to_linear_filter(profile)
        out[pos] = base + ',' + self.filter() + out[pos][len(original):]
        return self.wrap(out)

    def raw_command(self, ffmpeg, width, height, stats=False):
        return self.wrap([ffmpeg, '-hide_banner', '-nostdin', '-v', 'error',
            '-f', 'rawvideo', '-pixel_format', 'gbrpf32le',
            '-video_size', '{}x{}'.format(width,height), '-framerate', '1', '-i', 'pipe:0',
            '-vf', self.filter((width,height) if stats else None), '-frames:v', '1',
            '-pix_fmt', 'grayf32le' if stats else 'gbrpf32le', '-f', 'rawvideo', 'pipe:1'])


def tile_light_levels(data, width, height, white_nits):
    tw, th = (width+15)//16, (height+15)//16
    values = np.frombuffer(data, dtype='<f4').reshape((2, th, tw))
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Invalid GPU content-light measurements.')
    peak = float(values[0].max()) * white_nits
    average = float(values[1].sum(dtype=np.float64)) * white_nits / (width*height)
    if average > peak + .01 or peak > 10000.01:
        raise ValueError('Inconsistent GPU light measurements.')
    return peak, average


def _probe_run(command, payload, cancelled=None, register_process=None, timeout=30):
    """Cancellable bounded probe. Called on the export worker, never GUI thread."""
    from youtube_export_analysis import ExportCancelled
    if cancelled and cancelled():
        raise ExportCancelled()
    p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if register_process:
        register_process(p)
    start = time.monotonic()
    first = True
    try:
        while True:
            if cancelled and cancelled():
                raise ExportCancelled()
            try:
                output, error = p.communicate(input=payload if first else None, timeout=.2)
                if p.returncode:
                    raise ValueError(error.decode('utf-8',errors='replace')[-1000:])
                return output
            except subprocess.TimeoutExpired:
                first = False
                if time.monotonic()-start > timeout:
                    raise ValueError('GPU capability probe timed out.')
    finally:
        if p.poll() is None:
            p.kill()
        p.communicate()
        if register_process:
            register_process(None)


def choose_gpu_plan(ffmpeg, grade, white_nits, log=None, cancelled=None, register_process=None):
    """Require an executed pixel-parity test, not merely a listed GPU filter."""
    pixels = np.random.default_rng(17).uniform(0, 2, (3,16,16)).astype('<f4')
    pixels[:,0,:] = np.linspace(0,3,16)
    base = [ffmpeg,'-v','error','-f','rawvideo','-pixel_format','gbrpf32le',
            '-video_size','16x16','-i','pipe:0','-vf',
            build_creative_filter(grade,white_nits).lstrip(',') or 'null',
            '-frames:v','1','-pix_fmt','gbrpf32le','-f','rawvideo','pipe:1']
    expected = _probe_run(base, pixels.tobytes(), cancelled, register_process)
    errors = []
    for device in GPU_DEVICES:
        plan = GPUPlan(grade, white_nits, device)
        try:
            result = _probe_run(plan.raw_command(ffmpeg,16,16), pixels.tobytes(), cancelled, register_process)
            if len(result) != len(expected) or not np.allclose(
                    np.frombuffer(result,dtype='<f4'), np.frombuffer(expected,dtype='<f4'),
                    rtol=3e-4, atol=3e-5):
                raise ValueError('GPU grade differs from CPU reference.')
            stats = _probe_run(plan.raw_command(ffmpeg,16,16,stats=True), pixels.tobytes(), cancelled, register_process)
            from youtube_export_analysis import linear_frame_light_levels
            actual = tile_light_levels(stats,16,16,white_nits)
            target = linear_frame_light_levels(np.frombuffer(expected,dtype='<f4').reshape(3,16,16),white_nits)
            if not np.allclose(actual,target,rtol=3e-4,atol=.01):
                raise ValueError('GPU content-light reduction failed parity check.')
            if log:
                log('GPU colour processing: OpenCL {} (float grade and full-pixel analysis).'.format(device))
            return plan
        except (OSError, ValueError) as exc:
            plan.close()
            errors.append(str(exc).splitlines()[-1] if str(exc) else 'device unavailable')
        except BaseException:
            plan.close()
            raise
    if log:
        log('GPU processing unavailable; using CPU without changing export settings. ' + '; '.join(errors))
    return None
