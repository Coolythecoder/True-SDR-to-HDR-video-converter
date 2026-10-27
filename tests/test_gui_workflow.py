"""Regression coverage for dialogs, FFmpeg command construction and batch status."""

import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt5 import QtWidgets
    import SDR_to_HDR_cross_platform as converter_module
except ImportError:
    QtWidgets = None
    converter_module = None


@unittest.skipIf(converter_module is None, "Requires PyQt5 and OpenCV")
class GuiWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.widget = converter_module.SDRtoHDRConverter()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.widget.close()

    def test_output_selector_chooses_folder_in_single_mode(self):
        with mock.patch.object(QtWidgets.QFileDialog, "getExistingDirectory", return_value=str(self.root)):
            self.widget.select_output_file()
        self.assertEqual(self.widget.output_path.text(), str(self.root))

    def test_single_input_selector_returns_one_file(self):
        source = str(self.root / "video.mp4")
        with mock.patch.object(QtWidgets.QFileDialog, "getOpenFileName", return_value=(source, "")):
            self.widget.select_input_file()
        self.assertEqual(self.widget.input_path.text(), source)

    def test_missing_ffmpeg_is_reported_without_starting(self):
        source = self.root / "video.mp4"
        source.write_bytes(b"video")
        self.widget.input_path.setText(str(source))
        self.widget.output_path.setText(str(self.root))
        with mock.patch.object(converter_module, "find_ffmpeg", return_value=None):
            self.widget.start_conversion()
        self.assertIsNone(self.widget.conversion_thread)
        self.assertIn("FFmpeg not found", self.widget.status_label.text())

    def test_batch_builds_a_command_per_video(self):
        for name in ("first.mp4", "second.mov"):
            (self.root / name).write_bytes(b"video")
        output = self.root / "output"
        output.mkdir()
        self.widget.batch_mode.setChecked(True)
        self.widget.input_path.setText(str(self.root))
        self.widget.output_path.setText(str(output))
        with mock.patch.object(converter_module, "find_ffmpeg", return_value="ffmpeg"), \
             mock.patch.object(converter_module, "has_zscale", return_value=True), \
             mock.patch.object(converter_module, "find_ffprobe", return_value="ffprobe"), \
             mock.patch.object(converter_module, "probe_sdr_profile",
                               return_value=converter_module.DEFAULT_SDR_PROFILE), \
             mock.patch.object(converter_module.ConversionThread, "start"):
            self.widget.start_conversion()
        commands = self.widget.conversion_thread.ffmpeg_commands
        self.assertEqual(len(commands), 2)
        self.assertEqual({Path(c[c.index("-i") + 1]).name for c in commands},
                         {"first.mp4", "second.mov"})
        self.assertEqual({Path(c[-1]).name for c in commands},
                         {"first_HDR.mp4", "second_HDR.mp4"})

    def test_gpu_hevc_command_has_no_x265_specific_switches(self):
        self.widget.bit_depth.setCurrentText("10")
        self.widget.embed_metadata.setChecked(True)
        cmd = self.widget.build_ffmpeg_command("ffmpeg", "in.mp4", "out.mp4", use_gpu=True)
        self.assertEqual(cmd.count("-c:v"), 1)
        self.assertIn("hevc_nvenc", cmd)
        self.assertNotIn("-x265-params", cmd)

    def test_cpu_10bit_selects_x265(self):
        self.widget.bit_depth.setCurrentText("10")
        cmd = self.widget.build_ffmpeg_command("ffmpeg", "in.mp4", "out.mp4", use_gpu=False)
        self.assertIn("libx265", cmd)
        self.assertIn("yuv420p10le", cmd)


    def test_hdr_output_has_actual_pixel_conversion_and_colour_tags(self):
        self.assertEqual(self.widget.bit_depth.currentText(), "10")
        cmd = self.widget.build_ffmpeg_command(
            "ffmpeg", "in.mp4", "out.mp4", use_gpu=False
        )
        self.assertIn("-vf", cmd)
        filt = cmd[cmd.index("-vf") + 1]
        self.assertIn("zscale=rin=limited:min=bt709", filt)
        self.assertIn("format=gbrpf32le", filt)
        self.assertIn("t=smpte2084", filt)
        self.assertIn("npl=200", filt)
        self.assertEqual(cmd[cmd.index("-color_primaries") + 1], "bt2020")
        self.assertEqual(cmd[cmd.index("-color_trc") + 1], "smpte2084")
        self.assertIn("-x265-params", cmd)

    def test_8_bit_export_is_sdr_without_hdr_signalling(self):
        self.widget.bit_depth.setCurrentText("8")
        cmd = self.widget.build_ffmpeg_command("ffmpeg", "in.mp4", "out.mp4", use_gpu=False)
        self.assertNotIn("-vf", cmd)
        self.assertNotIn("-color_trc", cmd)
        self.assertNotIn("-x265-params", cmd)
        self.assertIn("libx264", cmd)
        self.assertFalse(self.widget.hdr_reference_nits.isEnabled())

    def test_hdr_reference_white_controls_mapping_and_mastering(self):
        self.widget.hdr_reference_nits.setValue(150)
        cmd = self.widget.build_ffmpeg_command("ffmpeg", "in.mp4", "out.mp4", use_gpu=False)
        self.assertIn("npl=150", cmd[cmd.index("-vf") + 1])
        self.assertIn("L(1500000,0)", cmd[cmd.index("-x265-params") + 1])

    def test_invalid_static_metadata_cannot_launch(self):
        self.widget.override_metadata.setChecked(True)
        self.widget.max_cll_input.setText("200")
        self.widget.max_fall_input.setText("999")
        with self.assertRaisesRegex(ValueError, "MaxFALL"):
            self.widget.build_ffmpeg_command("ffmpeg", "in.mp4", "out.mp4", use_gpu=False)

    def test_non_bt709_source_skips_unjustified_mastering_sei(self):
        profile = dict(converter_module.DEFAULT_SDR_PROFILE, primaries="bt2020")
        cmd = self.widget.build_ffmpeg_command(
            "ffmpeg", "in.mp4", "out.mp4", use_gpu=False, source_profile=profile
        )
        self.assertIn("pin=bt2020", cmd[cmd.index("-vf") + 1])
        self.assertNotIn("-x265-params", cmd)

    def test_rejects_hdr_source_metadata(self):
        source = self.root / "video.mp4"
        source.write_bytes(b"video")
        self.widget.input_path.setText(str(source))
        self.widget.output_path.setText(str(self.root))
        with mock.patch.object(converter_module, "find_ffmpeg", return_value="ffmpeg"), \
             mock.patch.object(converter_module, "has_zscale", return_value=True), \
             mock.patch.object(converter_module, "find_ffprobe", return_value="ffprobe"), \
             mock.patch.object(converter_module, "probe_sdr_profile",
                               side_effect=ValueError("Input is already HDR")):
            self.widget.start_conversion()
        self.assertIsNone(self.widget.conversion_thread)
        self.assertIn("already HDR", self.widget.status_label.text())

    def test_missing_zscale_is_reported(self):
        source = self.root / "video.mp4"
        source.write_bytes(b"video")
        self.widget.input_path.setText(str(source))
        self.widget.output_path.setText(str(self.root))
        with mock.patch.object(converter_module, "find_ffmpeg", return_value="ffmpeg"), \
             mock.patch.object(converter_module, "has_zscale", return_value=False):
            self.widget.start_conversion()
        self.assertIn("zscale", self.widget.status_label.text())
        self.assertIsNone(self.widget.conversion_thread)



    def test_youtube_button_forces_metadata_complete_cpu_export(self):
        video = self.root / "video.mp4"
        video.write_bytes(b"fake footage")
        self.widget.input_path.setText(str(video))
        self.widget.output_path.setText(str(self.root))
        self.widget.bit_depth.setCurrentText("8")
        self.widget.gpu_checkbox.setChecked(True)
        self.widget.audio_passthrough.setChecked(True)
        with mock.patch.object(converter_module, "find_ffmpeg", return_value="ffmpeg"), \
             mock.patch.object(converter_module, "find_ffprobe", return_value="ffprobe"), \
             mock.patch.object(converter_module, "has_zscale", return_value=True), \
             mock.patch.object(converter_module, "supports_libx265", return_value=True), \
             mock.patch.object(converter_module, "probe_sdr_profile",
                               return_value=converter_module.DEFAULT_SDR_PROFILE), \
             mock.patch.object(converter_module, "validate_youtube_source"), \
             mock.patch.object(converter_module.ConversionThread, "start"):
            self.widget.youtube_btn.click()
        self.assertIsNotNone(self.widget.conversion_thread)
        cmd = self.widget.conversion_thread.ffmpeg_commands[0]
        self.assertEqual(cmd[cmd.index("-c:v") + 1], "libx265")
        self.assertEqual(cmd[cmd.index("-profile:v") + 1], "main10")
        self.assertEqual(cmd[cmd.index("-c:a") + 1], "aac")
        self.assertIn("-x265-params", cmd)
        self.assertTrue(cmd[-1].endswith("video_YouTube_HDR.mp4"))
        self.assertIsNotNone(self.widget.conversion_thread.verify_output)
        self.assertFalse(self.widget.start_btn.isEnabled())
        self.assertFalse(self.widget.youtube_btn.isEnabled())

    def test_youtube_button_requires_ffprobe(self):
        video = self.root / "video.mp4"
        video.write_bytes(b"fake footage")
        self.widget.input_path.setText(str(video))
        self.widget.output_path.setText(str(self.root))
        with mock.patch.object(converter_module, "find_ffmpeg", return_value="ffmpeg"), \
             mock.patch.object(converter_module, "has_zscale", return_value=True), \
             mock.patch.object(converter_module, "find_ffprobe", return_value=None):
            self.widget.start_youtube_export()
        self.assertIsNone(self.widget.conversion_thread)
        self.assertIn("FFprobe", self.widget.status_label.text())

    def test_youtube_button_fails_without_libx265(self):
        video = self.root / "video.mp4"
        video.write_bytes(b"fake footage")
        self.widget.input_path.setText(str(video))
        self.widget.output_path.setText(str(self.root))
        with mock.patch.object(converter_module, "find_ffmpeg", return_value="ffmpeg"), \
             mock.patch.object(converter_module, "has_zscale", return_value=True), \
             mock.patch.object(converter_module, "find_ffprobe", return_value="ffprobe"), \
             mock.patch.object(converter_module, "supports_libx265", return_value=False):
            self.widget.start_youtube_export()
        self.assertIsNone(self.widget.conversion_thread)
        self.assertIn("libx265", self.widget.status_label.text())



@unittest.skipIf(converter_module is None, "Requires PyQt5 and OpenCV")
class WorkerTests(unittest.TestCase):
    @staticmethod
    def process(returncode, text=""):
        process = mock.Mock()
        process.stdout = io.StringIO(text)
        process.wait.return_value = returncode
        return process

    def test_worker_reports_ffmpeg_failure(self):
        worker = converter_module.ConversionThread([["ffmpeg", "-i", "input.mp4", "out.mp4"]])
        statuses = []
        worker.update_status.connect(statuses.append)
        with mock.patch.object(converter_module.subprocess, "Popen", return_value=self.process(1, "ffmpeg error\n")):
            worker.run()
        self.assertIn("1 failed", statuses[-1])

    def test_batch_continues_after_one_failed_file(self):
        commands = [["ffmpeg", "-i", "a.mp4", "a_HDR.mp4"],
                    ["ffmpeg", "-i", "b.mp4", "b_HDR.mp4"]]
        worker = converter_module.ConversionThread(commands)
        statuses = []
        worker.update_status.connect(statuses.append)
        with mock.patch.object(converter_module.subprocess, "Popen", side_effect=[self.process(1), self.process(0)]):
            worker.run()
        self.assertIn("1 succeeded, 1 failed", statuses[-1])

    def test_youtube_validation_failure_is_reported_as_failed_job(self):
        worker = converter_module.ConversionThread(
            [["ffmpeg", "-i", "video.mp4", "video_YouTube_HDR.mp4"]],
            verify_output=mock.Mock(side_effect=ValueError("PQ metadata missing")),
        )
        statuses = []
        logs = []
        worker.update_status.connect(statuses.append)
        worker.log_output.connect(logs.append)
        with mock.patch.object(
            converter_module.subprocess, "Popen", return_value=self.process(0)
        ):
            worker.run()
        self.assertIn("1 failed", statuses[-1])
        self.assertTrue(any("PQ metadata missing" in msg for msg in logs))

    def test_successful_post_encode_verification_is_reported(self):
        verify = mock.Mock(return_value=True)
        worker = converter_module.ConversionThread(
            [["ffmpeg", "-i", "video.mp4", "video_YouTube_HDR.mp4"]],
            verify_output=verify,
        )
        statuses = []
        worker.update_status.connect(statuses.append)
        with mock.patch.object(
            converter_module.subprocess, "Popen", return_value=self.process(0)
        ):
            worker.run()
        self.assertIn("1 succeeded", statuses[-1])
        verify.assert_called_once_with("video_YouTube_HDR.mp4")

    def test_cancel_terminates_process(self):
        worker = converter_module.ConversionThread([["ffmpeg", "-i", "a.mp4", "a_HDR.mp4"]])
        process = self.process(0)
        worker._process = process
        worker.cancel()
        process.terminate.assert_called_once()
        self.assertTrue(worker._cancel_requested)


if __name__ == "__main__":
    unittest.main()
