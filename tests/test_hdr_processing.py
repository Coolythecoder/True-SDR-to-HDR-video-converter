"""Unit tests for validated SDR colour profiles and FFmpeg filter generation."""

import json
import subprocess
import unittest
from unittest import mock

from hdr_processing import (
    DEFAULT_SDR_PROFILE, build_sdr_to_pq_filter, has_zscale,
    probe_sdr_profile, x265_sdr_mastering_params
)


class ColourProcessingTests(unittest.TestCase):
    def test_defaults_produce_real_pq_pixel_transform(self):
        pipeline = build_sdr_to_pq_filter(DEFAULT_SDR_PROFILE)
        self.assertIn("rin=limited:min=bt709:pin=bt709:tin=bt709", pipeline)
        self.assertIn("m=gbr:p=bt2020:t=linear", pipeline)
        self.assertIn("format=gbrpf32le", pipeline)
        self.assertIn("zscale=t=smpte2084:m=bt2020nc:p=bt2020", pipeline)
        self.assertIn("npl=200", pipeline)
        self.assertTrue(pipeline.endswith("format=yuv420p10le"))

    def test_nvenc_format_uses_p010le(self):
        pipeline = build_sdr_to_pq_filter(
            DEFAULT_SDR_PROFILE, 200, "p010le"
        )
        self.assertTrue(pipeline.endswith("format=p010le"))

    def test_full_range_source_is_preserved_before_conversion(self):
        source = dict(DEFAULT_SDR_PROFILE, range="full")
        self.assertIn("rin=full", build_sdr_to_pq_filter(source))

    def test_150_nit_white_changes_pq_not_primaries(self):
        pipeline = build_sdr_to_pq_filter(DEFAULT_SDR_PROFILE, 150)
        self.assertIn("npl=150", pipeline)
        self.assertIn("m=gbr:p=bt2020:t=linear", pipeline)

    def test_rejects_unsupported_profile_fields(self):
        for field in ("matrix", "primaries", "transfer", "range"):
            with self.subTest(field=field):
                profile = dict(DEFAULT_SDR_PROFILE)
                profile[field] = "bad;filter=inject"
                with self.assertRaisesRegex(ValueError, "Unsupported source"):
                    build_sdr_to_pq_filter(profile)

    def test_rejects_invalid_nits(self):
        with self.assertRaisesRegex(ValueError, "reference white"):
            build_sdr_to_pq_filter(DEFAULT_SDR_PROFILE, 1000)

    def test_rejects_non_10bit_hdr_pixel_format(self):
        with self.assertRaisesRegex(ValueError, "pixel format"):
            build_sdr_to_pq_filter(DEFAULT_SDR_PROFILE, pixel_format="yuv420p")

    def test_200_nit_st2086_is_bt709_source_primaries(self):
        params = x265_sdr_mastering_params(200)
        self.assertIn("G(15000,30000)B(7500,3000)R(32000,16500)", params)
        self.assertIn("L(2000000,0)", params)
        self.assertIn("max-cll=200,0", params)

    def test_rejects_invalid_cll_fall(self):
        for cll, fall in [(0, 0), (200, 300), (10001, 0), (500, -1)]:
            with self.subTest(cll=cll, fall=fall):
                with self.assertRaises(ValueError):
                    x265_sdr_mastering_params(200, cll, fall)

    def probe(self, stream):
        response = mock.Mock(returncode=0, stdout=json.dumps({"streams": [stream]}), stderr="")
        with mock.patch("hdr_processing.subprocess.run", return_value=response):
            return probe_sdr_profile("ffprobe", "sample.mp4")

    def test_probe_recognizes_full_range_bt709(self):
        profile = self.probe({
            "color_space": "bt709", "color_primaries": "bt709",
            "color_transfer": "bt709", "color_range": "pc"
        })
        self.assertEqual(profile, dict(DEFAULT_SDR_PROFILE, range="full"))

    def test_probe_handles_untagged_bt709_sdr(self):
        self.assertEqual(self.probe({}), DEFAULT_SDR_PROFILE)

    def test_probe_rejects_existing_pq_and_hlg_sources(self):
        for transfer in ("smpte2084", "arib-std-b67"):
            with self.subTest(transfer=transfer):
                with self.assertRaisesRegex(ValueError, "already HDR"):
                    self.probe({"color_transfer": transfer})

    def test_probe_rejects_unsupported_declared_colour(self):
        with self.assertRaisesRegex(ValueError, "unsupported color_primaries"):
            self.probe({"color_primaries": "smpte432"})

    def test_probe_supports_sd_bt601_and_bt2020_sdr(self):
        old = self.probe({
            "color_space": "smpte170m", "color_primaries": "smpte170m",
            "color_transfer": "smpte170m"
        })
        self.assertIn("min=smpte170m", build_sdr_to_pq_filter(old))
        wide = self.probe({
            "color_space": "bt2020nc", "color_primaries": "bt2020",
            "color_transfer": "bt2020-10"
        })
        self.assertIn("pin=bt2020", build_sdr_to_pq_filter(wide))

    def test_missing_ffprobe_has_explicit_fallback(self):
        self.assertEqual(probe_sdr_profile(None, "sample.mp4"), DEFAULT_SDR_PROFILE)

    def test_ffprobe_rejects_unreadable_input(self):
        result = mock.Mock(returncode=1, stdout="", stderr="unsupported video")
        with mock.patch("hdr_processing.subprocess.run", return_value=result):
            with self.assertRaisesRegex(ValueError, "Could not inspect"):
                probe_sdr_profile("ffprobe", "sample.mp4")

    def test_zscale_discovery(self):
        result = mock.Mock(returncode=0, stdout=" .SC zscale            V->V       Apply resizing.\n")
        with mock.patch("hdr_processing.subprocess.run", return_value=result):
            self.assertTrue(has_zscale("ffmpeg"))
        result.stdout = " T.C eq  V->V\n"
        with mock.patch("hdr_processing.subprocess.run", return_value=result):
            self.assertFalse(has_zscale("ffmpeg"))


if __name__ == "__main__":
    unittest.main()
