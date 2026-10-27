"""Regression tests for real HDR light-level analysis and EBU R128 audio."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from youtube_export_analysis import (
    ExportCancelled, _loudnorm_filter, apply_youtube_analysis,
    linear_frame_light_levels, measure_audio_loudness, measure_content_light,
)


class LinearLightTests(unittest.TestCase):
    def test_rec2020_uses_correct_gbr_plane_order(self):
        data = np.zeros((3, 2, 2), dtype=np.float32)
        data[2, 0, 0] = 1  # Red
        data[0, 0, 1] = 1  # Green
        data[1, 1, 0] = 1  # Blue
        peak, mean = linear_frame_light_levels(data, 200)
        self.assertAlmostEqual(peak, 200 * 0.678, delta=0.002)
        self.assertAlmostEqual(mean, 200 / 4, delta=0.002)

    def test_uniform_white_luminance(self):
        peak, mean = linear_frame_light_levels(
            np.ones((3, 2, 3), dtype=np.float32), 200
        )
        self.assertAlmostEqual(peak, 200.0, places=3)
        self.assertAlmostEqual(mean, 200.0, places=3)

    def test_negative_and_over_range_pixels_clamped(self):
        data = np.ones((3, 1, 2), dtype=np.float32)
        data[:, 0, 0] = -1
        data[:, 0, 1] = 100
        peak, mean = linear_frame_light_levels(data, 200)
        self.assertAlmostEqual(peak, 10000, delta=0.01)
        self.assertAlmostEqual(mean, 5000, delta=0.01)

    def test_nan_or_infinite_input_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                data = np.ones((3, 2, 2), dtype=np.float32)
                data[1, 0, 0] = value
                with self.assertRaisesRegex(ValueError, "Non-finite"):
                    linear_frame_light_levels(data, 200)


class AudioMeasurementTests(unittest.TestCase):
    @staticmethod
    def measurements(input_i="-26.9"):
        return {
            "input_i": input_i,
            "input_tp": "-10.5",
            "input_lra": "7.1",
            "input_thresh": "-37.2",
            "target_offset": "0.1",
        }

    def test_second_pass_contains_measured_data_and_stereo_downmix(self):
        output = _loudnorm_filter(self.measurements())
        self.assertTrue(output.startswith("aformat=channel_layouts=stereo,loudnorm="))
        self.assertIn("I=-16", output)
        self.assertIn("TP=-1.5", output)
        self.assertIn("measured_I=-26.9", output)
        self.assertIn("measured_TP=-10.5", output)
        self.assertIn("measured_LRA=7.1", output)
        self.assertIn("measured_thresh=-37.2", output)
        self.assertIn("offset=0.1", output)
        self.assertIn("linear=true", output)

    def test_invalid_statistics_rejected(self):
        stats = self.measurements()
        stats["target_offset"] = "NaN"
        with self.assertRaisesRegex(ValueError, "Invalid loudness"):
            _loudnorm_filter(stats)

    def test_silent_audio_is_not_amplified(self):
        stats = self.measurements("-inf")
        process = mock.Mock()
        process.returncode = 0
        process.communicate.return_value = ("", json.dumps(stats))
        process.poll.return_value = 0
        with mock.patch("youtube_export_analysis.subprocess.Popen", return_value=process):
            self.assertIsNone(measure_audio_loudness("ffmpeg", "in.mp4"))

    def test_reads_ffmpeg_loudnorm_json(self):
        stats = self.measurements()
        process = mock.Mock()
        process.returncode = 0
        process.communicate.return_value = (
            "", "some ffmpeg log lines\n[Parsed_loudnorm]\n" + json.dumps(stats) + "\n"
        )
        process.poll.return_value = 0
        with mock.patch("youtube_export_analysis.subprocess.Popen", return_value=process):
            self.assertEqual(measure_audio_loudness("ffmpeg", "in.mp4"), stats)

    def test_cancellation_before_audio_pass_does_not_spawn_ffmpeg(self):
        with mock.patch("youtube_export_analysis.subprocess.Popen") as fake:
            with self.assertRaises(ExportCancelled):
                measure_audio_loudness("ffmpeg", "file.mp4", cancelled=lambda: True)
            fake.assert_not_called()


class CommandEnrichmentTests(unittest.TestCase):
    def test_light_values_update_x265_metadata_and_audio_is_optional(self):
        cmd = [
            "ffmpeg", "-i", "input.mp4",
            "-x265-params", "max-cll=200,0",
            "-c:a", "aac", "-ar", "48000", "out.mp4",
        ]
        updated = apply_youtube_analysis(
            cmd, 200, light_levels=(160, 76, 50),
            audio_measurements=AudioMeasurementTests.measurements(),
        )
        self.assertIn("max-cll=160,76", updated[updated.index("-x265-params") + 1])
        self.assertLess(updated.index("-af"), updated.index("-c:a"))
        self.assertIn("measured_I=-26.9", updated[updated.index("-af") + 1])
        self.assertNotIn("-af", cmd)

    def test_without_options_command_is_not_modified(self):
        cmd = ["ffmpeg", "-x265-params", "whatever", "output.mp4"]
        updated = apply_youtube_analysis(cmd, 200)
        self.assertEqual(updated, cmd)
        self.assertIsNot(updated, cmd)


if __name__ == "__main__":
    unittest.main()
