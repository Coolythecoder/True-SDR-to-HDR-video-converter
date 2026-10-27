"""Reproduce legacy half-row copies and verify the actual GPU merge transport."""
import os
import shutil
import subprocess
import unittest

import numpy as np
from gpu_colour import GPUPlan
from gpu_float_transport import merge_float_planes

FFMPEG = shutil.which('ffmpeg')


def pattern(width, height, frames=1):
    """Distinct float GBR values everywhere, with negative and HDR headroom."""
    yy, xx = np.mgrid[:height, :width]
    planes = np.stack((.125 + xx / width + yy / height,
                       -.25 + 3 * xx / width + yy / height,
                       .5 + 18 * yy / height + xx / width)).astype('<f4')
    return np.stack([planes + np.float32(i / 8) for i in range(frames)])


def command(graph, width, height, frames=1):
    return [FFMPEG, '-v', 'error', '-f', 'rawvideo', '-pixel_format', 'gbrpf32le',
            '-video_size', '{}x{}'.format(width, height), '-framerate', '60',
            '-i', 'pipe:0', '-vf', graph, '-frames:v', str(frames),
            '-pix_fmt', 'gbrpf32le', '-f', 'rawvideo', 'pipe:1']


def extraction(prefix='transport'):
    return ('split=3[{0}r][{0}g][{0}b];'
            '[{0}r]extractplanes=r[{0}or];'
            '[{0}g]extractplanes=g[{0}og];'
            '[{0}b]extractplanes=b[{0}ob];').format(prefix)


class TransportRecipeTests(unittest.TestCase):
    def test_reassembly_is_shared_by_gpu_preview_and_export(self):
        plan = GPUPlan()
        try:
            self.assertIn(merge_float_planes('hgpu'), plan.filter())
            self.assertIn(merge_float_planes('pgpu'), plan.filter(prefix='pgpu'))
            # Statistics are already grayscale; they need no RGB merge.
            self.assertNotIn('mergeplanes', plan.filter(stats_size=(34, 18)))
        finally:
            plan.close()

    def test_no_quantisation_or_resizing_and_crop_is_left_aligned(self):
        graph = merge_float_planes('check')
        self.assertEqual(graph.count('hstack=inputs=2'), 3)
        self.assertIn('crop=w=iw/2:h=ih:x=0:y=0:exact=1', graph)
        self.assertNotIn('scale=', graph)
        self.assertNotIn('gray16', graph)
        self.assertNotIn('rgb24', graph)
        self.assertIn('color_trc=linear', graph)

    def test_invalid_labels_are_rejected(self):
        for label in ('', 'bad;graph', 'bad[graph]', '0label', None):
            with self.subTest(label=label), self.assertRaises(ValueError):
                merge_float_planes(label)


@unittest.skipUnless(FFMPEG, 'Requires FFmpeg')
class FloatTransportTests(unittest.TestCase):
    def roundtrip(self, width, height, frames=1):
        pixels = pattern(width, height, frames)
        graph = extraction() + merge_float_planes('transport')
        result = subprocess.run(command(graph, width, height, frames),
                                input=pixels.tobytes(), capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
        # Byte equality proves that even sign bits, float headroom and the
        # rightmost/bottommost pixels survive. It also checks frame count.
        self.assertEqual(len(result.stdout), pixels.nbytes)
        self.assertTrue(result.stdout == pixels.tobytes(),
                        'Float plane reassembly altered pixels at {}x{}'.format(width, height))

    def test_all_preview_sizes_and_partial_alignment_are_lossless(self):
        for width, height in ((34, 18), (480, 270), (640, 360), (960, 540)):
            with self.subTest(width=width, height=height):
                self.roundtrip(width, height, frames=2)

    def test_4k_float_pixels_are_bit_exact(self):
        self.roundtrip(3840, 2160)

    @unittest.skipUnless(os.environ.get('HDR_EXPECT_LEGACY_FLOAT_MERGE') == '1',
                         'Legacy-FFmpeg reproduction job only')
    def test_original_merge_reproduces_half_row_loss(self):
        width, height = 66, 34
        pixels = pattern(width, height)
        original = extraction() + (
            '[transportog][transportob][transportor]'
            'mergeplanes=0x001020:format=gbrpf32le')
        result = subprocess.run(command(original, width, height), input=pixels.tobytes(),
                                capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
        self.assertEqual(len(result.stdout), pixels.nbytes)
        actual = np.frombuffer(result.stdout, dtype='<f4').reshape(pixels.shape)
        np.testing.assert_array_equal(actual[..., :width//2], pixels[..., :width//2])
        self.assertFalse(np.array_equal(actual[..., width//2:], pixels[..., width//2:]),
                         'This FFmpeg no longer reproduces the legacy row-copy defect.')
        print('\nConfirmed: original merge copies the left half only; corrected transport follows.')
        self.roundtrip(width, height)


if __name__ == '__main__':
    unittest.main()
