"""Contract tests for the optional YouTube HDR export preset."""

import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hdr_processing import DEFAULT_SDR_PROFILE
from youtube_hdr_export import (
    YOUTUBE_CRF, YOUTUBE_SUFFIX, _mp4_moov_before_mdat, _mp4_has_edit_list,
    build_youtube_hdr_command, supports_libx265,
    validate_youtube_source, verify_youtube_hdr_file,
)


def _box(label, payload=b""):
    return struct.pack(">I4s", len(payload) + 8, label) + payload


class YouTubePresetTests(unittest.TestCase):
    def test_export_command_has_full_hdr10_flags_and_upload_audio(self):
        cmd = build_youtube_hdr_command(
            "ffmpeg", "input.mp4", "output.mp4", DEFAULT_SDR_PROFILE
        )
        self.assertEqual(cmd[cmd.index("-c:v") + 1], "libx265")
        self.assertEqual(cmd[cmd.index("-profile:v") + 1], "main10")
        self.assertEqual(cmd[cmd.index("-crf") + 1], "16")
        self.assertEqual(cmd[cmd.index("-color_primaries") + 1], "bt2020")
        self.assertEqual(cmd[cmd.index("-color_trc") + 1], "smpte2084")
        self.assertEqual(cmd[cmd.index("-colorspace") + 1], "bt2020nc")
        self.assertEqual(cmd[cmd.index("-pix_fmt") + 1], "yuv420p10le")
        self.assertEqual(cmd[cmd.index("-tag:v") + 1], "hvc1")
        self.assertIn("+faststart", cmd[cmd.index("-movflags") + 1])
        self.assertIn("+negative_cts_offsets", cmd[cmd.index("-movflags") + 1])
        self.assertEqual(cmd[cmd.index("-use_editlist") + 1], "0")
        self.assertEqual(cmd[cmd.index("-c:a") + 1], "aac")
        self.assertEqual(cmd[cmd.index("-ar") + 1], "48000")
        self.assertEqual(cmd[cmd.index("-ac") + 1], "2")
        self.assertEqual(cmd[cmd.index("-b:a") + 1], "384k")
        self.assertIn("format=gbrpf32le", cmd[cmd.index("-vf") + 1])
        self.assertIn("t=smpte2084", cmd[cmd.index("-vf") + 1])
        self.assertIn("npl=200", cmd[cmd.index("-vf") + 1])
        self.assertIn("master-display=", cmd[cmd.index("-x265-params") + 1])
        self.assertIn("max-cll=200,0", cmd[cmd.index("-x265-params") + 1])
        self.assertNotIn("hevc_nvenc", cmd)
        self.assertEqual(cmd[-1], "output.mp4")

    def test_youtube_profile_does_not_change_source_fps_or_resolution(self):
        cmd = build_youtube_hdr_command("ffmpeg", "in.mp4", "out.mp4", DEFAULT_SDR_PROFILE)
        self.assertNotIn("-r", cmd)
        self.assertNotIn("-s", cmd)
        self.assertNotIn("-vf", cmd[:cmd.index("-i")])

    def test_respects_hdr_white_and_validated_cll_override(self):
        cmd = build_youtube_hdr_command(
            "ffmpeg", "in.mp4", "out.mp4", DEFAULT_SDR_PROFILE,
            white_nits=150, crf=18, max_cll=220, max_fall=110
        )
        self.assertIn("npl=150", cmd[cmd.index("-vf") + 1])
        self.assertIn("max-cll=220,110", cmd[cmd.index("-x265-params") + 1])

    def test_rejects_high_crf_invalid_destination_and_non_bt709_source(self):
        with self.assertRaises(ValueError):
            build_youtube_hdr_command("ffmpeg", "a.mp4", "a.mkv", DEFAULT_SDR_PROFILE)
        with self.assertRaises(ValueError):
            build_youtube_hdr_command("ffmpeg", "a.mp4", "a.mp4", DEFAULT_SDR_PROFILE, crf=34)
        with self.assertRaises(ValueError):
            build_youtube_hdr_command(
                "ffmpeg", "a.mp4", "a.mp4",
                dict(DEFAULT_SDR_PROFILE, primaries="bt2020")
            )

    def test_encoder_discovery_requires_real_libx265(self):
        result = mock.Mock(returncode=0, stdout=" V....D hevc_nvenc Nvidia\n")
        with mock.patch("youtube_hdr_export.subprocess.run", return_value=result):
            self.assertFalse(supports_libx265("ffmpeg"))
        result.stdout += " V....D libx265 H.265\n"
        with mock.patch("youtube_hdr_export.subprocess.run", return_value=result):
            self.assertTrue(supports_libx265("ffmpeg"))

    def test_preflight_video_dimensions_framerate_and_progressive(self):
        source = {
            "codec_type": "video", "width": 1920, "height": 1080,
            "field_order": "progressive", "avg_frame_rate": "30000/1001",
            "r_frame_rate": "30000/1001",
        }
        with mock.patch("youtube_hdr_export._probe", return_value={"streams": [source]}):
            actual = validate_youtube_source("ffprobe", "movie.mp4", DEFAULT_SDR_PROFILE)
        self.assertEqual(actual["width"], 1920)

    def test_preflight_rejects_interlaced_undersized_and_unusual_fps(self):
        base = {
            "codec_type": "video", "width": 1920, "height": 1080,
            "field_order": "progressive", "avg_frame_rate": "30/1",
            "r_frame_rate": "30/1",
        }
        invalids = [
            ({"field_order": "tt"}, "Deinterlace"),
            ({"height": 480}, "at least 720"),
            ({"avg_frame_rate": "120/1", "r_frame_rate": "120/1"}, "frame rate"),
            ({"width": 1919}, "even video dimensions"),
        ]
        for altered, msg in invalids:
            with self.subTest(altered=altered):
                candidate = dict(base, **altered)
                with mock.patch("youtube_hdr_export._probe", return_value={"streams": [candidate]}):
                    with self.assertRaisesRegex(ValueError, msg):
                        validate_youtube_source("ffprobe", "in.mp4", DEFAULT_SDR_PROFILE)

    def test_preflight_rejects_wrong_primaries(self):
        with self.assertRaisesRegex(ValueError, "BT.709"):
            validate_youtube_source(
                "ffprobe", "in.mp4", dict(DEFAULT_SDR_PROFILE, primaries="smpte170m")
            )


class YouTubeVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.video = Path(self.temp.name) / "sample.mp4"
        self.video.write_bytes(
            _box(b"ftyp", b"iso6" + b"\x00" * 12) +
            _box(b"moov", _box(b"trak", _box(b"tkhd", b"x" * 80))) +
            _box(b"mdat", b"a" * 200)
        )
        self.probe_data = {"streams": [{
            "codec_type": "video", "codec_name": "hevc", "codec_tag_string": "hvc1",
            "profile": "Main 10", "pix_fmt": "yuv420p10le",
            "color_primaries": "bt2020", "color_transfer": "smpte2084",
            "color_space": "bt2020nc", "color_range": "tv",
            "width": 1280, "height": 720, "avg_frame_rate": "30/1",
        }, {
            "codec_type": "audio", "codec_name": "aac",
            "sample_rate": "48000", "channels": 2,
        }]}
        self.frames = {"frames": [{
            "side_data_list": [
                {"side_data_type": "Mastering display metadata"},
                {"side_data_type": "Content light level metadata"},
            ]
        }]}

    def test_moov_must_precede_video_data(self):
        self.assertTrue(_mp4_moov_before_mdat(self.video))
        self.video.write_bytes(_box(b"ftyp", b"iso6") + _box(b"mdat", b"a" * 200) + _box(b"moov", b"x" * 100))
        self.assertFalse(_mp4_moov_before_mdat(self.video))

    def test_absent_edit_list_is_accepted_and_present_edit_list_rejected(self):
        self.assertFalse(_mp4_has_edit_list(self.video))
        trak = _box(
            b"trak",
            _box(b"tkhd", b"x" * 80) +
            _box(b"edts", _box(b"elst", b"\x00" * 16))
        )
        self.video.write_bytes(
            _box(b"ftyp", b"iso6" + b"\x00" * 12) +
            _box(b"moov", trak) +
            _box(b"mdat", b"a" * 200)
        )
        self.assertTrue(_mp4_has_edit_list(self.video))
        with mock.patch("youtube_hdr_export._probe", return_value=self.probe_data):
            with self.assertRaisesRegex(ValueError, "edit lists"):
                verify_youtube_hdr_file("ffprobe", str(self.video))

    def test_output_verification_checks_hdr_sei_and_audio(self):
        proc = mock.Mock(returncode=0, stdout=json.dumps(self.frames), stderr="")
        with mock.patch("youtube_hdr_export._probe", return_value=self.probe_data), \
             mock.patch("youtube_hdr_export.subprocess.run", return_value=proc):
            self.assertTrue(verify_youtube_hdr_file("ffprobe", str(self.video)))

    def test_verification_fails_missing_static_hdr_metadata(self):
        proc = mock.Mock(returncode=0, stdout=json.dumps({"frames": [{}]}), stderr="")
        with mock.patch("youtube_hdr_export._probe", return_value=self.probe_data), \
             mock.patch("youtube_hdr_export.subprocess.run", return_value=proc):
            with self.assertRaisesRegex(ValueError, "missing HEVC HDR10 SEI"):
                verify_youtube_hdr_file("ffprobe", str(self.video))

    def test_verification_rejects_wrong_hevc_sample_entry(self):
        self.probe_data["streams"][0]["codec_tag_string"] = "hev1"
        with mock.patch("youtube_hdr_export._probe", return_value=self.probe_data):
            with self.assertRaisesRegex(ValueError, "codec_tag_string"):
                verify_youtube_hdr_file("ffprobe", str(self.video))

    def test_verification_fails_incorrect_colour_transfer(self):
        self.probe_data["streams"][0]["color_transfer"] = "bt709"
        with mock.patch("youtube_hdr_export._probe", return_value=self.probe_data):
            with self.assertRaisesRegex(ValueError, "color_transfer"):
                verify_youtube_hdr_file("ffprobe", str(self.video))

    def test_verification_fails_wrong_audio_sample_rate(self):
        self.probe_data["streams"][1]["sample_rate"] = "44100"
        with mock.patch("youtube_hdr_export._probe", return_value=self.probe_data):
            with self.assertRaisesRegex(ValueError, "audio"):
                verify_youtube_hdr_file("ffprobe", str(self.video))


if __name__ == "__main__":
    unittest.main()
