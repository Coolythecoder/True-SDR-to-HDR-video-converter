"""Execute FFmpeg's actual zscale pipeline against synthetic SDR colour samples.

These tests do not require a physical HDR monitor or a sample containing a person.
They verify the transfer/gamut numerically; real footage still needs visual QA.
"""

import array
import shutil
import subprocess
import unittest

from hdr_processing import DEFAULT_SDR_PROFILE, build_sdr_to_pq_filter, has_zscale


FFMPEG = shutil.which("ffmpeg")
SOURCE_TAGS = (
    "setparams=range=limited:color_primaries=bt709:"
    "color_trc=bt709:colorspace=bt709"
)
REVERSE = (
    "zscale=rin=limited:min=bt2020nc:pin=bt2020:tin=smpte2084:"
    "m=gbr:p=bt709:t=linear:r=full:npl=200,"
    "format=gbrpf32le,"
    "zscale=t=bt709:m=bt709:p=bt709:r=limited,"
    "format=yuv420p"
)


class FFmpegColourIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not FFMPEG or not has_zscale(FFMPEG):
            raise unittest.SkipTest("FFmpeg built with zscale/zimg is unavailable")

    def filtered_sample(self, color, filter_chain, fmt="yuv420p10le"):
        command = [
            FFMPEG, "-v", "error", "-f", "lavfi",
            "-i", "color=c={}:s=32x32:r=1:d=1".format(color),
            "-vf", SOURCE_TAGS + "," + filter_chain,
            "-frames:v", "1", "-pix_fmt", fmt,
            "-f", "rawvideo", "pipe:1",
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        return result.stdout

    def pq_yuv(self, color):
        data = self.filtered_sample(color, build_sdr_to_pq_filter(DEFAULT_SDR_PROFILE))
        yuv = array.array("H")
        yuv.frombytes(data)
        self.assertEqual(len(yuv), 32 * 32 * 3 // 2)
        return yuv[0], yuv[1024], yuv[1280]

    def test_white_reaches_200_nits_in_pq_not_1000(self):
        # 10-bit legal-range Y around 571 = 200 cd/m2 (SMPTE ST 2084).
        white = self.pq_yuv("white")
        self.assertTrue(567 <= white[0] <= 575, white)
        self.assertEqual(white[1:], (512, 512))

    def test_midgray_is_not_blowed_out(self):
        gray = self.pq_yuv("gray")
        self.assertTrue(422 <= gray[0] <= 434, gray)

    def test_black_stays_black(self):
        black = self.pq_yuv("black")
        self.assertTrue(63 <= black[0] <= 66, black)

    def test_skin_like_colour_roundtrip_preserves_components(self):
        original = self.filtered_sample("0xc08070", "null", "yuv420p")
        corrected = self.filtered_sample(
            "0xc08070",
            build_sdr_to_pq_filter(DEFAULT_SDR_PROFILE) + "," + REVERSE,
            "yuv420p",
        )
        # Constant skin-like sample; no chroma-edge resampling confusion.
        for index in (0, 1024, 1280):
            self.assertLessEqual(abs(original[index] - corrected[index]), 3)

    def test_brighter_skin_like_colour_roundtrip(self):
        original = self.filtered_sample("0xe4b29f", "null", "yuv420p")
        corrected = self.filtered_sample(
            "0xe4b29f",
            build_sdr_to_pq_filter(DEFAULT_SDR_PROFILE) + "," + REVERSE,
            "yuv420p",
        )
        for index in (0, 1024, 1280):
            self.assertLessEqual(abs(original[index] - corrected[index]), 3)


if __name__ == "__main__":
    unittest.main()
