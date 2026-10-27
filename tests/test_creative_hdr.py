"""Deterministic grading contract and real FFmpeg float-pixel tests."""
from dataclasses import replace
import shutil
import subprocess
import unittest
import numpy as np
from creative_hdr import CreativeHDRSettings, active_grade, build_creative_filter, creative_mastering_params


class CreativeSettingsTests(unittest.TestCase):
    def test_disabled_is_exact_noop(self):
        self.assertEqual(build_creative_filter(), '')
        self.assertEqual(build_creative_filter(CreativeHDRSettings()), '')

    def test_invalid_values_rejected(self):
        for field, value in [('peak_nits', 10001), ('strength', 2), ('knee', 1),
                             ('exposure', float('nan')), ('contrast', float('inf')),
                             ('saturation', '1;badfilter'), ('vibrance', True)]:
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    build_creative_filter(replace(CreativeHDRSettings(enabled=True), **{field: value}))

    def test_peak_must_not_be_below_reference_white(self):
        with self.assertRaises(ValueError):
            CreativeHDRSettings(enabled=True, peak_nits=200).validate(400)

    def test_unknown_settings_type_rejected(self):
        with self.assertRaises(ValueError):
            active_grade({'enabled': True})

    def test_creative_master_signals_virtual_bt2020_target_not_source_white(self):
        grade = CreativeHDRSettings(enabled=True, peak_nits=1000)
        value = creative_mastering_params(grade, 200)
        self.assertIn('G(8500,39850)B(6550,2300)R(35400,14600)', value)
        self.assertIn('L(10000000,0)', value)
        self.assertIn('max-cll=0,0', value)
        measured = creative_mastering_params(grade, 200, 650, 120)
        self.assertIn('L(10000000,0)', measured)
        self.assertIn('max-cll=650,120', measured)

    def test_bad_content_light_metadata_rejected(self):
        for cll, fall in [(1, 2), (-1, 0), (0, 100), (10001, 1)]:
            with self.assertRaises(ValueError):
                creative_mastering_params(CreativeHDRSettings(enabled=True), 200, cll, fall)


@unittest.skipUnless(shutil.which('ffmpeg'), 'Requires FFmpeg')
class CreativePixelTests(unittest.TestCase):
    @staticmethod
    def apply(data, grade):
        h, w = data.shape[1:]
        command = [shutil.which('ffmpeg'), '-v', 'error', '-f', 'rawvideo',
                   '-pix_fmt', 'gbrpf32le', '-s', '{}x{}'.format(w, h), '-i', 'pipe:0',
                   '-vf', build_creative_filter(grade)[1:] or 'null',
                   '-frames:v', '1', '-pix_fmt', 'gbrpf32le', '-f', 'rawvideo', 'pipe:1']
        result = subprocess.run(command, input=data.astype('<f4').tobytes(), capture_output=True, timeout=30)
        if result.returncode:
            raise AssertionError(result.stderr.decode(errors='replace'))
        return np.frombuffer(result.stdout, dtype='<f4').reshape(data.shape)

    def test_disabled_preserves_every_float(self):
        data = np.random.default_rng(1).uniform(0, 1.2, (3, 8, 16)).astype('<f4')
        np.testing.assert_array_equal(self.apply(data, CreativeHDRSettings()), data)

    def test_highlights_brighten_but_midtones_stay_unchanged(self):
        ramp = np.tile(np.linspace(0, 1, 257, dtype=np.float32), (3, 4, 1))
        result = self.apply(ramp, CreativeHDRSettings(enabled=True))
        np.testing.assert_allclose(result[:, :, :193], ramp[:, :, :193], atol=1e-6)
        self.assertAlmostEqual(float(result[0, 0, -1]) * 200, 400, delta=.02)
        self.assertTrue((np.diff(result[0, 0]) >= -1e-6).all())

    def test_no_neutral_colour_cast_or_raised_black_by_default(self):
        ramp = np.tile(np.linspace(0, 2, 256, dtype=np.float32), (3, 4, 1))
        result = self.apply(ramp, CreativeHDRSettings(enabled=True))
        np.testing.assert_array_equal(result[0], result[1])
        np.testing.assert_array_equal(result[1], result[2])
        self.assertEqual(float(result[0, 0, 0]), 0)
        self.assertLessEqual(float(result.max()), 3)

    def test_saturated_colours_and_all_controls_remain_finite_and_bounded(self):
        data = np.random.default_rng(2).uniform(-.1, 2, (3, 24, 32)).astype('<f4')
        settings = CreativeHDRSettings(enabled=True, peak_nits=1000, strength=.8,
                    exposure=.4, contrast=1.1, midtones=.3, shadows=.2, black_floor=.2,
                    saturation=1.5, vibrance=1, temperature=.8, tint=-.7)
        result = self.apply(data, settings)
        self.assertTrue(np.isfinite(result).all())
        self.assertGreaterEqual(float(result.min()), 0)
        self.assertLessEqual(float(result.max()), 5.00001)

    def test_saturation_zero_is_monochrome(self):
        data = np.random.default_rng(3).uniform(0, 1, (3, 8, 16)).astype('<f4')
        result = self.apply(data, CreativeHDRSettings(enabled=True, saturation=0))
        np.testing.assert_allclose(result[0], result[1], atol=2e-6)
        np.testing.assert_allclose(result[1], result[2], atol=2e-6)

    def test_temperature_and_tint_change_colour_not_only_tags(self):
        grey = np.full((3, 4, 8), .18, dtype='<f4')
        warm = self.apply(grey, CreativeHDRSettings(enabled=True, temperature=.5))
        self.assertGreater(float(warm[2, 0, 0]), float(warm[1, 0, 0]))
        magenta = self.apply(grey, CreativeHDRSettings(enabled=True, tint=.5))
        self.assertGreater(float(magenta[2, 0, 0]), float(magenta[0, 0, 0]))
        lum = .678*warm[0] + .0593*warm[1] + .2627*warm[2]
        np.testing.assert_allclose(lum, .18, atol=1e-5)

    def test_black_lift_is_explicit(self):
        black = np.zeros((3, 4, 8), dtype='<f4')
        result = self.apply(black, CreativeHDRSettings(enabled=True, black_floor=.5))
        np.testing.assert_allclose(result*200, .5, atol=1e-4)

    def test_hue_ratios_preserved_when_no_gamut_compression_needed(self):
        colour = np.empty((3, 4, 8), dtype='<f4')
        colour[0], colour[1], colour[2] = .90, .85, .95
        result = self.apply(colour, CreativeHDRSettings(enabled=True))
        np.testing.assert_allclose(result[2]/result[0], .95/.90, rtol=2e-6)
        np.testing.assert_allclose(result[1]/result[0], .85/.90, rtol=2e-6)

    def test_expansion_strength_zero_and_high_peak_preserve_in_range_image(self):
        data = np.random.default_rng(4).uniform(0, 1, (3, 8, 16)).astype('<f4')
        result = self.apply(data, CreativeHDRSettings(enabled=True, strength=0, peak_nits=1000))
        np.testing.assert_allclose(result, data, atol=2e-6)


if __name__ == '__main__':
    unittest.main()
