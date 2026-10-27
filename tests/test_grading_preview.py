"""Preview metadata, colour-recipe and real FFmpeg pixel regression tests."""
import shutil
import subprocess
import unittest
import numpy as np
from creative_hdr import CreativeHDRSettings, build_creative_filter
from grading_preview import source_from_probe, decode_command, render_command, viewing_filter


def metadata(**kwargs):
    video = dict(codec_type='video', width=1920, height=1080, duration='12.0',
                 avg_frame_rate='60/1', sample_aspect_ratio='1:1',
                 color_space='bt709', color_transfer='bt709', color_primaries='bt709', color_range='tv')
    video.update(kwargs)
    return {'streams': [video]}


class PreviewRecipes(unittest.TestCase):
    def test_full_and_limited_profiles_preserved(self):
        self.assertEqual(source_from_probe('test.mp4', metadata()).profile['range'], 'limited')
        self.assertEqual(source_from_probe('test.mp4', metadata(color_range='pc')).profile['range'], 'full')

    def test_missing_tags_are_disclosed(self):
        source = source_from_probe('test.mp4', metadata(color_range=None, color_transfer=None))
        self.assertIn('color_range', source.assumptions)
        self.assertIn('color_transfer', source.assumptions)

    def test_reject_hdr_and_unsupported_tags(self):
        for change in ({'color_transfer':'smpte2084'}, {'color_transfer':'arib-std-b67'},
                       {'color_primaries':'made-up'}, {'color_range':'wrong'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                source_from_probe('test.mp4', metadata(**change))

    def test_invalid_geometry_or_duration_fails(self):
        for change in ({'width':0}, {'height':40000}, {'duration':'N/A'}, {'duration':'nan'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                source_from_probe('test.mp4', metadata(**change))

    def test_rotated_and_anamorphic_dimensions_preserve_display_aspect(self):
        portrait = source_from_probe('rotated.mp4', metadata(side_data_list=[{'rotation':90}]))
        width, height = portrait.proxy_size()
        self.assertGreater(height, width)
        self.assertLess(abs(width / height - 9/16), .01)
        anamorphic = source_from_probe('anamorphic.mp4', metadata(width=1440, sample_aspect_ratio='4:3'))
        self.assertEqual(anamorphic.proxy_size(), (640, 360))

    def test_decode_is_read_only_single_frame_and_seeks_before_input(self):
        source = source_from_probe('spaces and ü;name.mp4', metadata())
        command = decode_command('ffmpeg', source, 5)
        self.assertLess(command.index('-ss'), command.index('-i'))
        self.assertEqual(command[command.index('-frames:v')+1], '1')
        self.assertEqual(command[-1], 'pipe:1')
        self.assertEqual(command[command.index('-i')+1], source.path)
        self.assertIn('rin=limited', command[command.index('-vf')+1])
        with self.assertRaises(ValueError):
            decode_command('ffmpeg', source, float('inf'))

    def test_render_reuses_exact_grade_and_same_view_on_both_halves(self):
        grade = CreativeHDRSettings(enabled=True, temperature=.4, saturation=1.2)
        command = render_command('ffmpeg', 64, 32, grade)
        graph = command[command.index('-filter_complex')+1]
        self.assertIn(build_creative_filter(grade).lstrip(','), graph)
        self.assertEqual(graph.count(viewing_filter()), 2)
        self.assertNotIn('libx265', command)
        self.assertNotIn('-y', command)
        self.assertEqual(command[-1], 'pipe:1')

    def test_display_scale_does_not_follow_grade_peak(self):
        for peak in (400, 600, 1000, 4000):
            command = render_command('ffmpeg', 64, 32, CreativeHDRSettings(enabled=True, peak_nits=peak))
            graph = command[command.index('-filter_complex')+1]
            self.assertEqual(graph.count('tonemap=mobius:param=0.3:desat=0:peak=5'), 2)


@unittest.skipUnless(shutil.which('ffmpeg'), 'Needs FFmpeg')
class PreviewPixels(unittest.TestCase):
    def render(self, grade, frame=None):
        if frame is None:
            frame = np.ones((3, 32, 64), dtype='<f4')
            frame[:, :, 32:] = .15
        result = subprocess.run(render_command(shutil.which('ffmpeg'), 64, 32, grade),
                                input=frame.tobytes(), capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
        self.assertEqual(len(result.stdout), 32 * 64 * 6)
        pair = np.frombuffer(result.stdout, np.uint8).reshape(32, 128, 3)
        return pair[:, :64], pair[:, 64:]

    def test_disabled_grade_is_pixel_identical(self):
        before, after = self.render(CreativeHDRSettings())
        np.testing.assert_array_equal(before, after)

    def test_highlights_visibly_change_without_hidden_autoexposure(self):
        before, after = self.render(CreativeHDRSettings(enabled=True))
        self.assertGreater(after[:, :30].mean(), before[:, :30].mean() + 5)
        np.testing.assert_array_equal(before[:, 34:], after[:, 34:])

    def test_display_transform_preserves_distinct_hdr_highlight_levels(self):
        values = []
        for light in (1, 2, 3, 5):
            before, _ = self.render(CreativeHDRSettings(), np.full((3,32,64), light, dtype='<f4'))
            values.append(int(before[0,0,0]))
        self.assertTrue(all(a < b for a,b in zip(values, values[1:])), values)
        self.assertEqual(values[-1], 255)

    def test_colour_control_affects_pixels(self):
        before, after = self.render(CreativeHDRSettings(enabled=True, temperature=.7))
        self.assertGreater(np.abs(after.astype(float)-before).mean(), 1)
        self.assertFalse(np.array_equal(after[:, 34:, 0], after[:, 34:, 2]))

    def test_saturation_zero_and_black_are_neutral(self):
        frame = np.zeros((3, 32, 64), dtype='<f4')
        frame[0, :, 32:] = .20
        frame[1, :, 32:] = .05
        frame[2, :, 32:] = .40
        _, after = self.render(CreativeHDRSettings(enabled=True, saturation=0), frame)
        self.assertLessEqual(int(after[:, :30].max()), 1)
        np.testing.assert_array_equal(after[:,:,0],after[:,:,1])
        np.testing.assert_array_equal(after[:,:,1],after[:,:,2])


if __name__ == '__main__':
    unittest.main()
