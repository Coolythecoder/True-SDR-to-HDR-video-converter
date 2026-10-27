"""Contract and real HDR10 encode tests for creative grading integration."""
import json
import shutil
import subprocess
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
import numpy as np
from creative_hdr import CreativeHDRSettings
from creative_hdr_pipeline import grade_command
from hdr_processing import DEFAULT_SDR_PROFILE, build_sdr_to_linear_filter, build_sdr_to_pq_filter, has_zscale
from youtube_hdr_export import build_youtube_hdr_command, supports_libx265, verify_youtube_hdr_file
from youtube_export_analysis import apply_youtube_analysis, measure_content_light


class GradePipelineTests(unittest.TestCase):
    def command(self):
        return build_youtube_hdr_command('ffmpeg', 'in.mp4', 'out.mp4', DEFAULT_SDR_PROFILE)

    def test_disabled_preserves_original_arguments(self):
        original = self.command()
        self.assertEqual(grade_command(original, DEFAULT_SDR_PROFILE, 200, CreativeHDRSettings()), original)
        self.assertEqual(build_sdr_to_pq_filter(DEFAULT_SDR_PROFILE), build_sdr_to_pq_filter(DEFAULT_SDR_PROFILE, grade=CreativeHDRSettings()))

    def test_scan_is_exact_prefix_of_export_filter(self):
        grade = CreativeHDRSettings(enabled=True, peak_nits=1000, saturation=1.1)
        scan = build_sdr_to_linear_filter(DEFAULT_SDR_PROFILE, grade=grade, white_nits=200)
        encode = build_sdr_to_pq_filter(DEFAULT_SDR_PROFILE, 200, grade=grade)
        self.assertTrue(encode.startswith(scan + ',zscale=t=smpte2084'))
        self.assertIn('geq=', scan)

    def test_metadata_replacement_keeps_grade_peak_and_colour_volume(self):
        grade = CreativeHDRSettings(enabled=True, peak_nits=1000)
        command = grade_command(self.command(), DEFAULT_SDR_PROFILE, 200, grade)
        prepared = apply_youtube_analysis(command, 200, light_levels=(750, 220, 100), grade=grade)
        params = prepared[prepared.index('-x265-params') + 1]
        self.assertIn('L(10000000,0)', params)
        self.assertIn('R(35400,14600)', params)
        self.assertIn('max-cll=750,220', params)
        self.assertEqual(prepared[-1], command[-1])
        self.assertEqual(prepared[prepared.index('-c:a'):], command[command.index('-c:a'):])

    def test_nvenc_does_not_receive_x265_parameters(self):
        original = self.command()
        index = original.index('-x265-params')
        del original[index:index + 2]
        original[original.index('-c:v') + 1] = 'hevc_nvenc'
        original[original.index('-pix_fmt') + 1] = 'p010le'
        grade = CreativeHDRSettings(enabled=True)
        result = grade_command(original, DEFAULT_SDR_PROFILE, 200, grade)
        result = apply_youtube_analysis(result, 200, light_levels=(400, 100, 10), grade=grade)
        self.assertNotIn('-x265-params', result)
        self.assertIn('geq=', result[result.index('-vf') + 1])

    def test_8bit_command_cannot_be_mislabelled_creative_hdr(self):
        original = self.command()
        original[original.index('-pix_fmt') + 1] = 'yuv420p'
        with self.assertRaisesRegex(ValueError, '10-bit'):
            grade_command(original, DEFAULT_SDR_PROFILE, 200, CreativeHDRSettings(enabled=True))


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'Requires FFmpeg/FFprobe')
class RealCreativeExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ffmpeg, cls.ffprobe = shutil.which('ffmpeg'), shutil.which('ffprobe')
        if not has_zscale(cls.ffmpeg) or not supports_libx265(cls.ffmpeg):
            raise unittest.SkipTest('Requires zscale and libx265')

    def test_full_and_limited_source_white_receive_equivalent_grade(self):
        grade = CreativeHDRSettings(enabled=True)
        results = []
        for value, video_range in [(235, 'limited'), (255, 'full')]:
            profile = dict(DEFAULT_SDR_PROFILE, range=video_range)
            raw = bytes([value]*16 + [128]*8)
            command = [self.ffmpeg, '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'yuv420p',
                       '-s', '4x4', '-i', 'pipe:0', '-vf', build_sdr_to_linear_filter(profile, grade=grade),
                       '-frames:v', '1', '-pix_fmt', 'gbrpf32le', '-f', 'rawvideo', 'pipe:1']
            result = subprocess.run(command, input=raw, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            results.append(np.frombuffer(result.stdout, dtype='<f4'))
        np.testing.assert_allclose(results[0], results[1], atol=.01)
        self.assertAlmostEqual(float(results[0].mean())*200, 400, delta=2)

    def test_actual_encoded_pq_is_brighter_and_contains_post_grade_sei(self):
        with tempfile.TemporaryDirectory() as directory:
            source = str(Path(directory)/'white.mp4')
            output = str(Path(directory)/'creative.mp4')
            creation = subprocess.run([
                self.ffmpeg, '-v', 'error', '-y', '-f', 'lavfi', '-i',
                'color=c=white:s=1280x720:r=24:d=0.125', '-c:v', 'libx264',
                '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', '-color_primaries', 'bt709',
                '-color_trc', 'bt709', '-colorspace', 'bt709', '-color_range', 'tv', source
            ], capture_output=True, text=True, timeout=60)
            self.assertEqual(creation.returncode, 0, creation.stderr)
            grade = CreativeHDRSettings(enabled=True)
            before = measure_content_light(self.ffmpeg, source, DEFAULT_SDR_PROFILE, 1280, 720, 200)
            after = measure_content_light(self.ffmpeg, source, DEFAULT_SDR_PROFILE, 1280, 720, 200, grade=grade)
            self.assertAlmostEqual(before[0], 200, delta=2)
            self.assertAlmostEqual(after[0], 400, delta=5)
            self.assertEqual(before[2], after[2])
            command = grade_command(build_youtube_hdr_command(self.ffmpeg, source, output, DEFAULT_SDR_PROFILE), DEFAULT_SDR_PROFILE, 200, grade)
            command = apply_youtube_analysis(command, 200, light_levels=after, grade=grade)
            encoded = subprocess.run(command, capture_output=True, text=True, timeout=120)
            self.assertEqual(encoded.returncode, 0, encoded.stderr[-3000:])
            self.assertTrue(verify_youtube_hdr_file(self.ffprobe, output))
            probed = subprocess.run([self.ffprobe, '-v', 'error', '-select_streams', 'v:0',
                                    '-read_intervals', '%+#3', '-show_frames', '-of', 'json', output],
                                   capture_output=True, text=True, timeout=30)
            self.assertEqual(probed.returncode, 0, probed.stderr)
            side = {item['side_data_type']: item for frame in json.loads(probed.stdout)['frames'] for item in frame.get('side_data_list', [])}
            self.assertEqual(side['Content light level metadata']['max_content'], after[0])
            self.assertEqual(side['Content light level metadata']['max_average'], after[1])
            master = side['Mastering display metadata']
            self.assertEqual(Fraction(master['max_luminance']), 600)
            self.assertAlmostEqual(float(Fraction(master['red_x'])), .708, places=4)
            # Decode actual PQ samples to relative linear light, not tags alone.
            decoded = subprocess.run([
                self.ffmpeg, '-v', 'error', '-i', output, '-vf',
                'zscale=rin=limited:min=bt2020nc:pin=bt2020:tin=smpte2084:t=linear:m=gbr:p=bt2020:r=full:npl=200,format=gbrpf32le',
                '-frames:v', '1', '-pix_fmt', 'gbrpf32le', '-f', 'rawvideo', 'pipe:1'
            ], capture_output=True, timeout=30)
            self.assertEqual(decoded.returncode, 0, decoded.stderr)
            pixels = np.frombuffer(decoded.stdout, dtype='<f4')
            self.assertAlmostEqual(float(pixels.mean()) * 200, 400, delta=8)


if __name__ == '__main__':
    unittest.main()
