"""Optional strict NVIDIA GPU-resident custom HDR export.

Bulk video pixels stay on CUDA: NVDEC -> float colour/grade -> PQ/P010 -> NVENC.
CPU orchestration, packet/timing validation, audio and MP4 muxing remain. No
silent CPU fallback. Existing FFmpeg/OpenCL and YouTube paths are not modified.
"""
from dataclasses import dataclass
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import time

import numpy as np
from creative_hdr import CreativeHDRSettings
from cuda_colour import cuda_source
from youtube_export_analysis import ExportCancelled


@dataclass(frozen=True)
class GPUOnlySettings:
    grade: CreativeHDRSettings = CreativeHDRSettings()
    white_nits: int = 200
    qp: int = 23
    copy_audio: bool = False
    device: int = 0

    def validate(self):
        self.grade.validate(self.white_nits)
        if type(self.qp) is not int or not 0 <= self.qp <= 51:
            raise ValueError('NVENC QP must be 0 to 51.')
        if type(self.white_nits) is not int or type(self.device) is not int or self.device < 0:
            raise ValueError('Invalid CUDA device or reference white.')
        if type(self.copy_audio) is not bool:
            raise ValueError('Invalid audio setting.')
        return self


@dataclass(frozen=True)
class SourceInfo:
    width: int
    height: int
    bits: int
    fps: Fraction
    time_base: Fraction
    full_range: bool
    centred: bool


def validate_source(data):
    """Conservative v1 contract; unsupported media uses the existing mode."""
    videos = [s for s in data.get('streams', []) if s.get('codec_type') == 'video']
    if len(videos) != 1:
        raise ValueError('GPU-only mode requires exactly one video stream.')
    v = videos[0]
    if v.get('codec_name') not in ('h264', 'hevc'):
        raise ValueError('GPU-only v1 supports H.264/HEVC inputs. Use normal custom export for this codec.')
    for key in ('color_space', 'color_primaries', 'color_transfer'):
        if v.get(key) != 'bt709':
            raise ValueError('GPU-only v1 requires explicitly tagged BT.709 SDR; unsupported ' + key)
    formats = {'yuv420p': 8, 'yuvj420p': 8, 'yuv420p10le': 10}
    if v.get('pix_fmt') not in formats:
        raise ValueError('GPU-only v1 requires 8/10-bit 4:2:0 SDR input.')
    width, height = int(v['width']), int(v['height'])
    if width % 2 or height % 2 or not (16 <= width <= 7680 and 16 <= height <= 4320):
        raise ValueError('GPU-only input must have even dimensions, from 16x16 through 7680x4320.')
    if v.get('field_order') != 'progressive':
        raise ValueError('GPU-only v1 requires an explicitly progressive source; use normal export otherwise.')
    if v.get('sample_aspect_ratio', '1:1') not in ('1:1', 'N/A', '0:1'):
        raise ValueError('GPU-only v1 requires square pixels; use normal export for anamorphic input.')
    rotations = [v.get('tags', {}).get('rotate', 0)]
    rotations += [s.get('rotation', 0) for s in v.get('side_data_list', [])]
    if any(float(x) % 360 for x in rotations):
        raise ValueError('GPU-only v1 does not rotate frames; use normal export for this source.')
    fps = Fraction(v.get('avg_frame_rate', '0/1'))
    tb = Fraction(v.get('time_base', '0/1'))
    if not 1 <= fps <= 240 or tb <= 0:
        raise ValueError('GPU-only v1 requires a valid 1-240 fps time base.')
    rng = v.get('color_range')
    if rng not in ('pc', 'tv'):
        raise ValueError('GPU-only mode requires an explicit full/limited source range.')
    chroma = v.get('chroma_location', 'unspecified')
    if chroma not in ('left', 'center', 'unspecified'):
        raise ValueError('GPU-only v1 supports left/centre chroma locations only.')
    return SourceInfo(width, height, formats[v['pix_fmt']], fps, tb, rng == 'pc', chroma == 'center')


def validate_timestamps(timestamps, info):
    """Check EVERY packet PTS. Do not turn VFR/offset/missing frames into CFR."""
    pts = sorted(timestamps)
    if not pts:
        raise ValueError('No video packets found.')
    if abs(pts[0] * info.time_base) > info.time_base:
        raise ValueError('GPU-only v1 requires video starting at zero; use normal export for offset timelines.')
    # Exact rational comparison; tolerate one input clock tick of rounding.
    ticks_per_frame = 1 / (info.fps * info.time_base)
    if ticks_per_frame < 2:
        raise ValueError('Source timestamp precision is too low for strict GPU-only CFR validation.')
    for i, value in enumerate(pts):
        if i and value == pts[i-1]:
            raise ValueError('Duplicate video timestamp; use normal custom export.')
        if abs(Fraction(value - pts[0]) - i * ticks_per_frame) > 1:
            raise ValueError('Variable/discontinuous frame timing; GPU-only v1 will not silently retime it.')
    return len(pts)


def _run(command, cancelled, register, timeout=120):
    """Cancellable subprocess capture; commands are argument arrays, never shell."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    register(process)
    start = time.monotonic()
    try:
        while True:
            if cancelled():
                raise ExportCancelled()
            if time.monotonic() - start > timeout:
                raise ValueError('GPU-only validation/mux command timed out.')
            try:
                out, err = process.communicate(timeout=.2)
                break
            except subprocess.TimeoutExpired:
                continue
        if cancelled():
            raise ExportCancelled()
        if process.returncode:
            raise ValueError(err.decode('utf-8', errors='replace')[-2000:] or 'Media command failed.')
        return out
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate()
        register(None)


def sei_messages(settings, cll, fall):
    """Big-endian HEVC SEI payloads, using the existing virtual-master policy."""
    if not 0 <= fall <= cll <= 10000:
        raise ValueError('Invalid measured HDR light levels.')
    if settings.grade.enabled:
        primaries = (8500, 39850, 6550, 2300, 35400, 14600)
        peak = settings.grade.peak_nits
    else:
        primaries = (15000, 30000, 7500, 3000, 32000, 16500)
        peak = settings.white_nits
    mastering = struct.pack('>8H2I', *primaries, 15635, 16450, round(peak*10000), 0)
    light = struct.pack('>2H', cll, fall)
    return [({'sei_type': 137}, list(mastering)), ({'sei_type': 144}, list(light))]


def encoder_parameters(info, settings):
    # Native NVENC settings are explicit, not claimed byte-identical to FFmpeg.
    return dict(codec='hevc', gpu_id=settings.device, preset='P4',
                tuning_info='high_quality', rc='constqp', constqp=settings.qp,
                fps=round(float(info.fps)), bf=0, gop=max(1, round(float(info.fps)*2)),
                repeatspspps=1, timinginfo={'num_unit_in_ticks': info.fps.denominator,
                                           'timescale': info.fps.numerator})


def mux_command(ffmpeg, elementary, source, destination, info, settings):
    fps = str(info.fps)
    return [ffmpeg, '-hide_banner', '-nostdin', '-v', 'error', '-y',
            '-copyts', '-fflags', '+genpts', '-r', fps, '-i', str(elementary), '-i', str(source),
            '-map', '0:v:0', '-map', '1:a:0?', '-map_metadata', '-1', '-c:v', 'copy',
            '-bsf:v', 'hevc_metadata=video_full_range_flag=0:colour_primaries=9:'
            'transfer_characteristics=16:matrix_coefficients=9:chroma_sample_loc_type=1:'
            'tick_rate={}:num_ticks_poc_diff_one=1'.format(fps),
            '-color_primaries', 'bt2020', '-color_trc', 'smpte2084',
            '-colorspace', 'bt2020nc', '-color_range', 'tv', '-tag:v', 'hvc1',
            '-c:a', 'copy' if settings.copy_audio else 'aac'] + (
            [] if settings.copy_audio else ['-b:a', '128k']) + [
            '-avoid_negative_ts', 'disabled', '-movflags', '+faststart', str(destination)]


_DLL_HANDLES = []


def load_cuda():
    """Lazy imports keep the normal application usable without CUDA packages."""
    import sys
    root = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent))
    bundled = root / 'cuda'
    if bundled.is_dir():
        os.environ['CUDA_PATH'] = str(bundled)
        os.environ['PATH'] = str(bundled / 'bin') + os.pathsep + os.environ.get('PATH', '')
    if os.name == 'nt' and hasattr(os, 'add_dll_directory'):
        paths = [root / 'PyNvVideoCodec', bundled / 'bin']
        paths += [Path(sys.prefix) / 'Lib/site-packages/nvidia' / pkg / 'bin'
                  for pkg in ('cuda_runtime', 'cuda_nvrtc')]
        for path in paths:
            if path.is_dir():
                _DLL_HANDLES.append(os.add_dll_directory(str(path)))
    try:
        import cupy as cp
        import PyNvVideoCodec as nvc
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError('No CUDA GPU found.')
        return cp, nvc
    except Exception as exc:
        raise ValueError('GPU-only mode needs NVIDIA drivers, CuPy and PyNvVideoCodec. '
                         'Use the GPU Windows build or requirements-gpu-only.txt. '
                         'No CPU fallback was used. ' + str(exc)[-600:]) from exc


class EncoderFrame:
    """Keep the P010 owner alive and expose correctly byte-strided CAI planes."""
    def __init__(self, buffer, height, width):
        self.buffer = buffer
        self.planes = [buffer[:height].reshape(height, width, 1),
                       buffer[height:].reshape(height//2, width//2, 2)]

    def cuda(self):
        return self.planes


class CUDAProcessor:
    def __init__(self, cp, info, settings):
        self.cp, self.info, self.settings = cp, info, settings
        self.module = cp.RawModule(code=cuda_source(settings.grade, settings.white_nits),
                                   options=('--std=c++11', '--fmad=false'))
        self.convert = self.module.get_function('convert8' if info.bits == 8 else 'convert16')
        self.pack = self.module.get_function('pack_p010')
        w, h = info.width, info.height
        self.colour = cp.empty((3, h, w), dtype=cp.float32)
        self.light = cp.empty((h, w), dtype=cp.float32)
        self.output = cp.empty((h*3//2, w), dtype=cp.uint16)
        self.encoded_frame = EncoderFrame(self.output, h, w)
        self.peak = cp.zeros((), dtype=cp.float64)
        self.fall = cp.zeros((), dtype=cp.float64)

    def process(self, frame, encode=False, linear_probe=False):
        cp, info = self.cp, self.info
        planes = frame.cuda()
        if len(planes) != 2:
            raise ValueError('NVDEC did not return two semiplanar CUDA planes.')
        y, uv = (cp.asarray(p) for p in planes)
        dtype = cp.uint8 if info.bits == 8 else cp.uint16
        w, h = info.width, info.height
        if y.dtype != dtype or uv.dtype != dtype or y.shape[0] != h or uv.shape[0] != h//2:
            raise ValueError('Unexpected NVDEC surface layout/bit depth. CPU fallback is disabled.')
        if y.strides[0] < w*y.itemsize or uv.strides[0] < w*uv.itemsize:
            raise ValueError('Truncated NVDEC pitch.')
        args = (y, uv, np.int32(y.strides[0]//y.itemsize), np.int32(uv.strides[0]//uv.itemsize),
                np.int32(w), np.int32(h), np.int32(info.full_range), np.int32(info.centred),
                np.int32(info.bits), np.float32(self.settings.white_nits), np.int32(2 if linear_probe else (1 if encode else 0)),
                self.colour, self.light)
        self.convert(((w+15)//16, (h+15)//16), (16, 16), args)
        if linear_probe:
            return self.colour
        if encode:
            self.pack(((w//2+15)//16, (h//2+15)//16), (16, 16),
                      (self.colour, self.output, np.int32(w), np.int32(h)))
            return self.encoded_frame
        self.peak = cp.maximum(self.peak, cp.max(self.light))
        self.fall = cp.maximum(self.fall, cp.mean(self.light, dtype=cp.float64))


def verify_output(ffprobe, path, info, count, cll, fall, cancelled, register):
    data = json.loads(_run([ffprobe, '-v', 'error', '-select_streams', 'v:0', '-count_packets',
                           '-show_streams', '-of', 'json', str(path)], cancelled, register, 600))
    v = data['streams'][0]
    required = {'codec_name': 'hevc', 'profile': 'Main 10', 'pix_fmt': 'yuv420p10le',
                'color_primaries': 'bt2020', 'color_transfer': 'smpte2084',
                'color_space': 'bt2020nc', 'color_range': 'tv', 'codec_tag_string': 'hvc1'}
    if any(v.get(k) != val for k, val in required.items()):
        raise ValueError('GPU-only output failed HEVC Main 10 / PQ / BT.2020 verification.')
    if (v.get('width'), v.get('height')) != (info.width, info.height):
        raise ValueError('GPU-only output dimensions changed.')
    if Fraction(v['avg_frame_rate']) != info.fps or int(v['nb_read_packets']) != count:
        raise ValueError('GPU-only output frame timing/count changed.')
    # One decoded verification frame is not bulk processing. No CPU fallback.
    first = json.loads(_run([ffprobe, '-v', 'error', '-select_streams', 'v:0',
                            '-read_intervals', '%+#1', '-show_frames', '-of', 'json', str(path)],
                           cancelled, register))
    side = first.get('frames', [{}])[0].get('side_data_list', [])
    light = [s for s in side if s.get('side_data_type') == 'Content light level metadata']
    master = [s for s in side if s.get('side_data_type') == 'Mastering display metadata']
    if not master or not light or (light[0].get('max_content'), light[0].get('max_average')) != (cll, fall):
        raise ValueError('GPU-only output is missing the measured HDR10 metadata.')


def validate_cuda_backend(cp, nvc, ffmpeg, ffprobe, source_info, settings, stream, cancelled, register, log):
    """Tiny startup reference/codec tests, never a CPU fallback for the movie."""
    from hdr_processing import build_sdr_to_linear_filter
    info = SourceInfo(128, 64, source_info.bits, source_info.fps, source_info.time_base,
                      source_info.full_range, source_info.centred)
    processor = CUDAProcessor(cp, info, settings)
    mult = 4 if info.bits == 10 else 1
    dtype = np.uint16 if info.bits == 10 else np.uint8
    profile = dict(matrix='bt709', primaries='bt709', transfer='bt709',
                   range='full' if info.full_range else 'limited')
    log('GPU-only: running small CPU-reference and native NVENC self-tests before the full video.')
    with tempfile.TemporaryDirectory(prefix='hdr_cuda_check_') as td:
        input_file = Path(td) / 'probe.yuv'
        for y, u, v in ((16,128,128), (126,128,128), (235,128,128), (96,96,176)):
            if cancelled():
                raise ExportCancelled()
            host = np.empty((96,128),dtype=dtype)
            host[:64] = y*mult
            host[64:,0::2] = u*mult
            host[64:,1::2] = v*mult
            if info.bits == 10:
                host <<= 6
            input_file.write_bytes(host.tobytes())
            device = cp.asarray(host)
            frame = EncoderFrame(device,64,128)
            actual = processor.process(frame,linear_probe=True).get()
            raw = _run([ffmpeg,'-v','error','-f','rawvideo','-pixel_format',
                        'p010le' if info.bits==10 else 'nv12','-video_size','128x64',
                        '-i',str(input_file),'-vf',build_sdr_to_linear_filter(profile,settings.grade,settings.white_nits),
                        '-frames:v','1','-pix_fmt','gbrpf32le','-f','rawvideo','pipe:1'],cancelled,register)
            if len(raw) != 128*64*12:
                raise ValueError('Incomplete GPU-only CPU-reference self-test.')
            reference = np.frombuffer(raw,dtype='<f4').reshape(3,64,128)[[2,0,1]]
            if not np.allclose(actual,reference,rtol=3e-4,atol=3e-5):
                raise ValueError('CUDA colour self-test differs from the existing reference; no export started.')
        surface = processor.process(frame,encode=True)
        stream.synchronize()
        encoder = nvc.CreateEncoder(128,64,'P010',False,cudastream=stream.ptr,**encoder_parameters(info,settings))
        if info.fps.denominator != 1:
            p = encoder.GetEncodeReconfigureParams()
            p.frameRateNum,p.frameRateDen = info.fps.numerator,info.fps.denominator
            encoder.Reconfigure(p)
        raw_path, out_path = Path(td)/'probe.hevc', Path(td)/'probe.mp4'
        metadata = sei_messages(settings,100,50)
        with raw_path.open('wb') as output:
            output.write(bytes(encoder.Encode(surface,0,metadata)))
            output.write(bytes(encoder.Encode(surface,0,metadata)))
            output.write(bytes(encoder.EndEncode()))
        # Reuse the video-only part of the mux recipe; there is no audio input.
        cmd = mux_command(ffmpeg,raw_path,raw_path,out_path,info,settings)
        second = cmd.index('-i',cmd.index('-i')+1)
        del cmd[second:second+2]
        mapping = cmd.index('1:a:0?')
        del cmd[mapping-1:mapping+1]
        _run(cmd,cancelled,register)
        verify_output(ffprobe,out_path,info,2,100,50,cancelled,register)
    log('GPU-only self-tests passed: colour reference, native P010 encode, timing and HDR10 metadata.')


def export_gpu_only(ffmpeg, ffprobe, source, destination, settings, log=print,
                    cancelled=lambda: False, register=lambda p: None):
    """Two GPU passes followed by lossless video remux; publish only on success."""
    settings.validate()
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.suffix.lower() != '.mp4':
        raise ValueError('GPU-only output must be a distinct MP4 file.')
    if not source.is_file() or not destination.parent.is_dir():
        raise ValueError('Source or destination folder is unavailable.')
    original_stat = (source.stat().st_size, source.stat().st_mtime_ns)
    probe = json.loads(_run([ffprobe, '-v', 'error', '-show_streams', '-of', 'json', str(source)], cancelled, register))
    info = validate_source(probe)
    log('GPU-only: checking every packet timestamp before processing (no video decode).')
    packet_text = _run([ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                        'packet=pts', '-of', 'csv=p=0', str(source)], cancelled, register, 600)
    try:
        timestamps = [int(line.split(',')[0]) for line in packet_text.decode().splitlines() if line.strip()]
    except ValueError as exc:
        raise ValueError('Source contains missing/non-numeric packet timestamps.') from exc
    expected = validate_timestamps(timestamps, info)
    del packet_text, timestamps
    cp, nvc = load_cuda()
    cp.cuda.Device(settings.device).use()
    stream = cp.cuda.Stream(non_blocking=False)
    with stream:
        validate_cuda_backend(cp, nvc, ffmpeg, ffprobe, info, settings, stream, cancelled, register, log)
        processor = CUDAProcessor(cp, info, settings)
        name = cp.cuda.runtime.getDeviceProperties(settings.device)['name']
        if isinstance(name, bytes):
            name = name.decode(errors='replace')
        log('GPU-only backend: {} | NVDEC -> CUDA colour/grade/PQ/P010 -> NVENC. '
            'CPU audio, mux and verification only; no CPU image-processing fallback.'.format(name))
        log('GPU-only native NVENC: Main 10, P4, constant QP {}, {} fps; centred chroma and ordered dither.'.format(settings.qp, info.fps))
        w, h = info.width, info.height
        encoder = nvc.CreateEncoder(w, h, 'P010', False, cudastream=stream.ptr,
                                    **encoder_parameters(info, settings))
        if info.fps.denominator != 1:
            p = encoder.GetEncodeReconfigureParams()
            p.frameRateNum, p.frameRateDen = info.fps.numerator, info.fps.denominator
            encoder.Reconfigure(p)

        def frames():
            decoder = nvc.SimpleDecoder(str(source), gpu_id=settings.device, cuda_stream=stream.ptr,
                                        use_device_memory=True, output_color_type=nvc.OutputColorType.NATIVE)
            while True:
                if cancelled():
                    raise ExportCancelled()
                batch = decoder.get_batch_frames(1)
                if not batch:
                    break
                stream.synchronize()
                for frame in batch:
                    yield frame
                    # Native surfaces may be reused on the next decode call.
                    stream.synchronize()

        count, started = 0, time.monotonic()
        for frame in frames():
            processor.process(frame)
            count += 1
            if count % 120 == 0:
                log('Analysed {} video frames for HDR light levels (GPU-only, {:.1f} fps).'.format(count, count/max(.001,time.monotonic()-started)))
        if count != expected:
            raise ValueError('NVDEC frame count differs from packet timing scan; output was not created.')
        peak, average = float(processor.peak.get()), float(processor.fall.get())
        if not math.isfinite(peak) or not math.isfinite(average):
            raise ValueError('Non-finite GPU light measurements; export stopped.')
        cll, fall = min(10000, math.ceil(peak)), min(10000, math.ceil(average))
        fall = min(fall, cll)
        log('Measured HDR light levels: MaxCLL={} nits, MaxFALL={} nits ({} frames).'.format(cll, fall, count))
        sei = sei_messages(settings, cll, fall)
        with tempfile.TemporaryDirectory(prefix='.hdr_gpu_', dir=destination.parent) as td:
            raw = Path(td) / 'video.hevc'
            complete = Path(td) / 'complete.mp4'
            encoded, started = 0, time.monotonic()
            with raw.open('wb') as output:
                for frame in frames():
                    surface = processor.process(frame, encode=True)
                    stream.synchronize()
                    output.write(bytes(encoder.Encode(surface, 0, sei if encoded == 0 else [])))
                    encoded += 1
                    if encoded % 120 == 0:
                        log('frame={} fps={:.1f} GPU-only NVENC'.format(encoded, encoded/max(.001,time.monotonic()-started)))
                output.write(bytes(encoder.EndEncode()))
            stream.synchronize()
            if encoded != count or original_stat != (source.stat().st_size, source.stat().st_mtime_ns):
                raise ValueError('Source or frame count changed between passes; export stopped.')
            if cancelled():
                raise ExportCancelled()
            log('GPU-only: muxing the encoded video and selected audio; no video re-encoding.')
            _run(mux_command(ffmpeg, raw, source, complete, info, settings), cancelled, register, 3600)
            verify_output(ffprobe, complete, info, count, cll, fall, cancelled, register)
            if cancelled():
                raise ExportCancelled()
            os.replace(complete, destination)
            log('Verified GPU-only HDR output: ' + destination.name)
