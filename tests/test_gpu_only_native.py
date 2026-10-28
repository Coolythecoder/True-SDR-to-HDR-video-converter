"""Opt-in real NVIDIA regression tests: HDR_TEST_CUDA=1 python -m unittest
discover -s tests -p test_gpu_only_native.py -v

These tests require the pinned CUDA dependencies, NVIDIA GPU and FFmpeg.
"""
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import warnings

import numpy as np

from gpu_only_export import (EncoderFrame, GPUOnlySettings, SourceInfo,
                             encoder_parameters, export_gpu_only, load_cuda)


@unittest.skipUnless(os.environ.get('HDR_TEST_CUDA') == '1', 'Set HDR_TEST_CUDA=1 for real NVIDIA tests')
class NativeGPUExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Source installs with the runtime wheels must discover CUDA before
        # CuPy imports, without masking the warning from the reported failure.
        with warnings.catch_warnings():
            warnings.filterwarnings('error', message='CUDA path could not be detected.*')
            cls.cp, cls.nvc = load_cuda()
        cls.cp.cuda.Device(0).use()

    def command(self, args):
        result = subprocess.run(args, capture_output=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace')[-3000:])
        return result.stdout

    def test_p010_native_encode_preserves_luma_and_both_chroma_words(self):
        from gpu_only_export import _write_encoded_packets
        cp, nvc = self.cp, self.nvc
        width, height = 256, 256
        host = np.empty((height * 3 // 2, width), dtype=np.uint16)
        host[:height, :width//2] = 200 << 6
        host[:height, width//2:] = 700 << 6
        host[height:, 0::2] = 350 << 6
        host[height:, 1::2] = 650 << 6
        frame = EncoderFrame(cp.asarray(host), height, width)
        cp.cuda.get_current_stream().synchronize()
        info = SourceInfo(width, height, 10, Fraction(24), Fraction(1,24000), False, False)
        encoder = nvc.CreateEncoder(width, height, 'P010', False,
                                    cudacontext=int(cp.cuda.driver.ctxGetCurrent()),
                                    **encoder_parameters(info, GPUOnlySettings(qp=0)))
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / 'probe.hevc'
            with raw.open('wb') as output:
                for _ in range(3):
                    _write_encoded_packets(output, encoder.Encode(frame))
                _write_encoded_packets(output, encoder.EndEncode())
            decoded = self.command(['ffmpeg', '-v', 'error', '-i', str(raw), '-frames:v', '1',
                                    '-pix_fmt', 'p010le', '-f', 'rawvideo', 'pipe:1'])
        actual = np.frombuffer(decoded, dtype='<u2').reshape(host.shape).astype(np.int32) >> 6
        expected = host.astype(np.int32) >> 6
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2)

    def test_short_export_preserves_frames_timing_colour_and_hdr_metadata(self):
        for bits, fps in ((8, Fraction(144)), (10, Fraction(24000,1001))):
            with self.subTest(bits=bits, fps=fps), tempfile.TemporaryDirectory() as td:
                source, target = Path(td) / 'source.mp4', Path(td) / 'hdr.mp4'
                # A non-aligned width also exercises native surface byte pitch.
                codec = (['-c:v', 'libx264'] if bits == 8 else
                         ['-c:v', 'libx265', '-x265-params', 'pools=1:frame-threads=1'])
                self.command(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                              'testsrc2=size=258x194:rate=' + str(fps), '-frames:v', '6',
                              '-pix_fmt', 'yuv420p' if bits == 8 else 'yuv420p10le'] + codec +
                              ['-preset', 'ultrafast', '-bf', '0',
                              '-color_primaries', 'bt709', '-color_trc', 'bt709',
                              '-colorspace', 'bt709', '-color_range', 'tv', str(source)])
                logs = []
                export_gpu_only('ffmpeg', 'ffprobe', source, target, GPUOnlySettings(), log=logs.append)
                data = json.loads(self.command(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                                                '-show_streams', '-show_packets', '-of', 'json', str(target)]))
                stream = data['streams'][0]
                self.assertEqual((stream['width'], stream['height']), (258,194))
                self.assertEqual(stream['profile'], 'Main 10')
                self.assertEqual(stream['color_transfer'], 'smpte2084')
                self.assertEqual(stream['color_primaries'], 'bt2020')
                self.assertEqual(Fraction(stream['avg_frame_rate']), fps)
                tb = Fraction(stream['time_base'])
                self.assertEqual([int(p['pts']) * tb for p in data['packets']], [i / fps for i in range(6)])
                self.assertTrue(any('Verified GPU-only HDR output' in line for line in logs))
