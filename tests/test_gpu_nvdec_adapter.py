"""Native decoder descriptor regression checks and opt-in sample validation."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from gpu_only_export import _NVDECPlane, load_cuda


class Plane:
    def __init__(self, interface):
        self.__cuda_array_interface__ = interface


class NVDECAdapterTests(unittest.TestCase):
    def test_native_p010_geometry_uses_byte_strides_and_retains_owner(self):
        frame = object()
        for shape, strides, expected in (([194,258,1], [258,1,1], (516,2,2)),
                                         ([97,129,2], [258,2,1], (516,4,2))):
            original = dict(version=3, shape=shape, strides=strides,
                            typestr='|u2', stream=0, data=(123456, False), gpuIdx=0)
            plane = Plane(original)
            adapter = _NVDECPlane(plane, frame)
            actual = adapter.__cuda_array_interface__
            self.assertEqual(actual['shape'], tuple(shape))
            self.assertEqual(actual['strides'], expected)
            self.assertEqual(actual['typestr'], np.dtype('uint16').str)
            self.assertEqual(actual['data'], original['data'])
            self.assertNotIn('stream', actual)
            self.assertIs(adapter.frame, frame)
            self.assertIs(adapter.plane, plane)
            self.assertIs(original['strides'], strides)
            self.assertEqual(original['typestr'], '|u2')

    def test_nv12_and_conforming_p010_preserve_byte_pitch(self):
        for typestr, strides in (('|u1', [320,1,1]), ('<u2', [640,2,2])):
            original = dict(version=3, shape=[194,258,1], strides=strides,
                            typestr=typestr, stream=123, data=(123456, False))
            actual = _NVDECPlane(Plane(original), object()).__cuda_array_interface__
            self.assertEqual(actual['strides'], tuple(strides))
            self.assertEqual(actual['stream'], 123)


@unittest.skipUnless(os.environ.get('HDR_TEST_CUDA') == '1', 'Set HDR_TEST_CUDA=1 for real NVIDIA tests')
class NativeNVDECSampleTests(unittest.TestCase):
    def test_nv12_and_p010_samples_match_software_decode(self):
        cp, nvc = load_cuda()
        cp.cuda.Device(0).use()

        def decode_native(path):
            with cp.cuda.Stream(non_blocking=True) as stream:
                decoder = nvc.SimpleDecoder(str(path), gpu_id=0,
                    cuda_context=int(cp.cuda.driver.ctxGetCurrent()), cuda_stream=stream.ptr,
                    use_device_memory=True, output_color_type=nvc.OutputColorType.NATIVE)
                frame = decoder.get_batch_frames(1)[0]
                stream.synchronize()
                native = frame.cuda()
                planes = [cp.asarray(_NVDECPlane(plane, frame)) for plane in native]
                for actual, original in zip(planes, native):
                    self.assertEqual(actual.data.ptr, original.__cuda_array_interface__['data'][0])
                y, uv = [plane.get() for plane in planes]
                return np.concatenate((y.reshape(-1), uv.reshape(-1)))

        for bits in (8,10):
            with self.subTest(bits=bits), tempfile.TemporaryDirectory() as td:
                path = Path(td) / 'source.mp4'
                codec = (['-c:v', 'libx264'] if bits == 8 else
                         ['-c:v', 'libx265', '-x265-params', 'pools=1:frame-threads=1'])
                subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                    'testsrc2=size=258x194:rate=24', '-frames:v', '1',
                    '-pix_fmt', 'yuv420p' if bits == 8 else 'yuv420p10le'] + codec +
                    ['-preset', 'ultrafast', str(path)],
                    check=True, capture_output=True, timeout=120)
                raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path),
                    '-frames:v', '1', '-pix_fmt', 'nv12' if bits == 8 else 'p010le',
                    '-f', 'rawvideo', 'pipe:1'], check=True, capture_output=True, timeout=120).stdout
                expected = np.frombuffer(raw, dtype=np.uint8 if bits == 8 else '<u2')
                np.testing.assert_array_equal(decode_native(path), expected)


if __name__ == '__main__':
    unittest.main()
