"""Real FFmpeg YouTube HDR export test; skipped if codecs/zscale are unavailable."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from hdr_processing import DEFAULT_SDR_PROFILE, has_zscale
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

            self.assertEqual(
                validate_youtube_source(FFPROBE, original, DEFAULT_SDR_PROFILE)["width"], 1280
            )
            cmd = build_youtube_hdr_command(
                FFMPEG, original, upload, DEFAULT_SDR_PROFILE
            )
            encoded = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120
            )
            self.assertEqual(encoded.returncode, 0, encoded.stderr[-4000:])
            self.assertTrue(verify_youtube_hdr_file(FFPROBE, upload))


if __name__ == "__main__":
    unittest.main()
