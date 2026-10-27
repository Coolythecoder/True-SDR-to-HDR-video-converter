"""Real FFmpeg YouTube HDR export test; skipped if codecs/zscale are unavailable."""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from hdr_processing import DEFAULT_SDR_PROFILE, has_zscale
from youtube_export_analysis import (
    apply_youtube_analysis, measure_audio_loudness, measure_content_light,
)
from youtube_hdr_export import (
    build_youtube_hdr_command,
    supports_libx265,
    validate_youtube_source,
    verify_youtube_hdr_file,
)


FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


class RealYouTubeExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not FFMPEG or not FFPROBE:
            raise unittest.SkipTest("FFmpeg/FFprobe are not installed")
        if not has_zscale(FFMPEG) or not supports_libx265(FFMPEG):
            raise unittest.SkipTest("FFmpeg lacks zscale or libx265")

    def test_encode_and_verify_real_hevc_hdr10_mp4(self):
        with tempfile.TemporaryDirectory() as temp:
            original = str(Path(temp) / "source_sdr.mp4")
            upload = str(Path(temp) / "youtube_hdr.mp4")

            source_cmd = [
                FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=c=0xc08070:s=1280x720:r=24:d=0.25",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=0.25",
                "-t", "0.25", "-c:v", "libx264", "-preset", "ultrafast",
                "-pix_fmt", "yuv420p", "-color_primaries", "bt709",
                "-color_trc", "bt709", "-colorspace", "bt709",
                "-c:a", "aac", "-ar", "48000", original,
            ]
            source = subprocess.run(
                source_cmd, capture_output=True, text=True, timeout=60
            )
            self.assertEqual(source.returncode, 0, source.stderr)

            source_info = validate_youtube_source(FFPROBE, original, DEFAULT_SDR_PROFILE)
            self.assertEqual(source_info["width"], 1280)
            self.assertTrue(source_info["has_audio"])
            levels = measure_content_light(
                FFMPEG, original, DEFAULT_SDR_PROFILE, 1280, 720, 200
            )
            self.assertGreater(levels[0], 1)
            self.assertGreaterEqual(levels[0], levels[1])
            self.assertGreaterEqual(levels[2], 5)
            command = build_youtube_hdr_command(
                FFMPEG, original, upload, DEFAULT_SDR_PROFILE
            )
            cmd = apply_youtube_analysis(command, 200, light_levels=levels)
            self.assertIn(
                "max-cll={},{}".format(levels[0], levels[1]),
                cmd[cmd.index("-x265-params") + 1],
            )
            encoded = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120
            )
            self.assertEqual(encoded.returncode, 0, encoded.stderr[-4000:])
            self.assertTrue(verify_youtube_hdr_file(FFPROBE, upload))

            metadata = subprocess.run(
                [FFPROBE, "-v", "error", "-select_streams", "v:0",
                 "-read_intervals", "%+#1", "-show_frames",
                 "-of", "json", upload],
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(metadata.returncode, 0, metadata.stderr)
            frames = json.loads(metadata.stdout).get("frames", [])
            light = [
                item
                for frame in frames
                for item in frame.get("side_data_list", [])
                if item.get("side_data_type") == "Content light level metadata"
            ]
            self.assertTrue(light)
            self.assertEqual(light[0]["max_content"], levels[0])
            self.assertEqual(light[0]["max_average"], levels[1])

    def test_two_pass_loudness_measured_and_applied(self):
        with tempfile.TemporaryDirectory() as temp:
            source = str(Path(temp) / "sine.m4a")
            output = str(Path(temp) / "normalized.m4a")
            make_audio = subprocess.run(
                [FFMPEG, "-v", "error", "-y", "-f", "lavfi",
                 "-i", "sine=frequency=440:sample_rate=48000:duration=3",
                 "-c:a", "aac", "-b:a", "192k", source],
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(make_audio.returncode, 0, make_audio.stderr)
            measured = measure_audio_loudness(FFMPEG, source)
            self.assertIsNotNone(measured)
            self.assertLess(float(measured["input_i"]), -17)
            skeleton = [
                FFMPEG, "-v", "error", "-y", "-i", source,
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                "-ac", "2", output,
            ]
            command = apply_youtube_analysis(
                skeleton, 200, audio_measurements=measured
            )
            normalized = subprocess.run(
                command, capture_output=True, text=True, timeout=45,
            )
            self.assertEqual(normalized.returncode, 0, normalized.stderr)
            normalized_measurement = measure_audio_loudness(FFMPEG, output)
            self.assertIsNotNone(normalized_measurement)
            self.assertAlmostEqual(
                float(normalized_measurement["input_i"]), -16, delta=1.2
            )
            self.assertLessEqual(
                float(normalized_measurement["input_tp"]), -1.0
            )


if __name__ == "__main__":
    unittest.main()
