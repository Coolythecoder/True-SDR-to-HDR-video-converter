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

    def test_cancel_terminates_process(self):
        worker = converter_module.ConversionThread([["ffmpeg", "-i", "a.mp4", "a_HDR.mp4"]])
        process = self.process(0)
        worker._process = process
        worker.cancel()
        process.terminate.assert_called_once()
        self.assertTrue(worker._cancel_requested)


if __name__ == "__main__":
    unittest.main()
