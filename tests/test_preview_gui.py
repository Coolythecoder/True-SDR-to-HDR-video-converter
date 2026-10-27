"""Qt interaction, cancellation, race prevention and real live-preview tests."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PyQt5 import QtCore, QtGui, QtTest, QtWidgets
from creative_hdr import CreativeHDRSettings
from grading_preview import source_from_probe
from preview_widget import PreviewEngine
from SDR_to_HDR_cross_platform import SDRtoHDRConverter
from workspace_ui import set_exporting


def wait_until(predicate, timeout=10):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        QtTest.QTest.qWait(20)
        if predicate():
            return True
    return False


def image_hash(image):
    return hashlib.sha256(image.bits().asstring(image.byteCount())).hexdigest()


class WorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.window = SDRtoHDRConverter()

    def tearDown(self):
        self.window.close()
        QtWidgets.QApplication.processEvents()

    def test_preview_and_actions_present_with_grouped_controls(self):
        w = self.window
        self.assertIsNotNone(w.preview)
        self.assertEqual(w.controls_tabs.count(), 3)
        self.assertEqual(w.creative_panel.tabs.count(), 3)
        self.assertEqual(len(w.creative_panel.sliders), 13)
        self.assertFalse(w.creative_panel.isChecked())
        self.assertEqual(w.youtube_quality.currentText(), 'High')
        self.assertIsNotNone(w.findChild(QtWidgets.QScrollArea))

    def test_slider_spin_sync_and_one_signal_for_preset(self):
        panel = self.window.creative_panel
        spy = QtTest.QSignalSpy(panel.settingsChanged)
        panel.apply_preset('Creative 600 nits')
        self.assertEqual(len(spy), 1)
        panel.sliders['saturation'].setValue(75)
        self.assertEqual(panel.controls['saturation'].value(), .75)
        self.assertEqual(panel.presets.currentText(), 'Custom')
        panel.controls['exposure'].setValue(1.25)
        self.assertEqual(panel.sliders['exposure'].value(), 125)
        panel.reset_neutral()
        self.assertFalse(panel.isChecked())
        self.assertEqual(panel.controls['saturation'].value(), 1)

    def test_grade_changes_debounce_live_preview(self):
        engine = self.window.preview.engine
        revision = engine.revision
        self.window.creative_panel.apply_preset('Gentle 400 nits')
        self.assertGreater(engine.revision, revision)
        self.assertTrue(engine.debounce.isActive())
        self.assertEqual(engine.grade.peak_nits, 400)

    def test_batch_does_not_automatically_pass_a_directory_to_ffmpeg(self):
        w = self.window
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(w.preview, 'load_source') as load:
            w.batch_mode.setChecked(True)
            w.input_path.setText(tmp)
            QtTest.QTest.qWait(500)
            load.assert_not_called()
        self.assertIn('Batch', w.preview.info.text())

    def test_export_pauses_preview_and_reenables_controls(self):
        w = self.window
        set_exporting(w, True)
        self.assertTrue(w.preview.engine.paused)
        self.assertFalse(w.grade_page.isEnabled())
        self.assertFalse(w.source_group.isEnabled())
        self.assertTrue(w.cancel_btn.isEnabled())
        self.assertTrue(w.log_output.isEnabled())
        set_exporting(w, False)
        self.assertTrue(w.grade_page.isEnabled())
        self.assertFalse(w.preview.engine.paused)

    def test_source_change_clears_preceding_preview(self):
        w = self.window
        image = QtGui.QImage(32, 16, QtGui.QImage.Format_RGB32)
        image.fill(QtCore.Qt.red)
        w.preview.canvas.set_frames(image, image)
        w.input_path.setText('different-file.mp4')
        self.assertIsNone(w.preview.canvas.before)

    def test_display_mode_and_preview_resolution_do_not_change_export(self):
        w = self.window
        before = w.build_ffmpeg_command('ffmpeg', 'a.mp4', 'b.mp4', False)
        w.preview.mode.setCurrentText('After grade')
        w.preview.quality.setCurrentIndex(0)
        after = w.build_ffmpeg_command('ffmpeg', 'a.mp4', 'b.mp4', False)
        self.assertEqual(before, after)

    def test_missing_tools_error_is_actionable(self):
        with tempfile.NamedTemporaryFile(suffix='.mp4') as file:
            with mock.patch('preview_widget.find_ffmpeg', return_value=None):
                self.window.preview.load_source(file.name)
        self.assertIn('FFmpeg', self.window.preview.status.text())


class AsyncPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.engine = PreviewEngine()

    def tearDown(self):
        self.engine.shutdown()
        QtWidgets.QApplication.processEvents()

    def test_superseded_process_cannot_publish_stale_data(self):
        values = []
        self.engine._launch([sys.executable, '-c', 'import time; time.sleep(.5); print("old")'], 'render', values.append)
        QtTest.QTest.qWait(30)
        self.engine.cancel()
        self.engine._launch([sys.executable, '-c', 'print("new")'], 'render', values.append)
        self.assertTrue(wait_until(lambda: len(values) == 1))
        QtTest.QTest.qWait(550)
        self.assertEqual(values, [b'new\r\n'] if os.name == 'nt' else [b'new\n'])

    def test_failed_start_is_reported_without_blocking(self):
        failures = []
        self.engine.failed.connect(failures.append)
        self.engine._launch(['missing-ffmpeg-79172639', '-version'], 'probe', lambda _: None)
        self.assertTrue(wait_until(lambda: bool(failures), 3))
        self.assertIn('Cannot start', failures[0])

    def test_output_size_limit_is_enforced(self):
        failures, results = [], []
        self.engine.failed.connect(failures.append)
        self.engine._launch([sys.executable, '-c', 'print("x"*100000)'], 'probe', results.append, limit=100)
        self.assertTrue(wait_until(lambda: bool(failures)))
        self.assertFalse(results)
        self.assertIn('safety limit', failures[0])

    def test_regrade_uses_cached_source_frame(self):
        engine = self.engine
        engine.source = source_from_probe('dummy.mp4', {'streams': [{
            'codec_type':'video','width':640,'height':360,'duration':'2',
            'avg_frame_rate':'30/1','color_space':'bt709','color_transfer':'bt709',
            'color_primaries':'bt709','color_range':'tv'}]})
        key = engine._key()
        engine.cache[key] = b'cached linear frame'
        engine.ffmpeg = 'ffmpeg'
        with mock.patch.object(engine, '_launch') as launch:
            engine.request(CreativeHDRSettings(enabled=True))
            engine._schedule()
        self.assertEqual(launch.call_args.args[1], 'render')
        self.assertEqual(launch.call_args.kwargs['payload'], b'cached linear frame')

    def test_outdated_render_result_is_discarded(self):
        engine = self.engine
        engine.source = source_from_probe('dummy.mp4', {'streams': [{
            'codec_type':'video','width':640,'height':360,'duration':'2','avg_frame_rate':'30/1'}]})
        engine.ffmpeg = 'ffmpeg'
        engine.cache[engine._key()] = b'cached'
        outputs = []
        engine.frameReady.connect(lambda *args: outputs.append(args))
        with mock.patch.object(engine, '_launch') as launch:
            engine._schedule()
            stale_callback = launch.call_args.args[2]
            engine.request(CreativeHDRSettings(enabled=True, saturation=.2))
            stale_callback(b'old image should not even be decoded')
        self.assertEqual(outputs, [])


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'Needs FFmpeg and FFprobe')
class LivePreviewIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_real_file_live_regrade_seek_and_shutdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'Colour sample ; Ω.mp4')
            result = subprocess.run([
                shutil.which('ffmpeg'), '-v', 'error', '-f', 'lavfi',
                '-i', 'testsrc2=s=640x360:r=24:d=1', '-c:v', 'libx264',
                '-preset', 'ultrafast', '-pix_fmt', 'yuv420p',
                '-color_primaries', 'bt709', '-color_trc', 'bt709',
                '-colorspace', 'bt709', '-color_range', 'tv', path,
            ], capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            window = SDRtoHDRConverter()
            window.resize(1280, 800)
            window.show()
            frames, failures = [], []
            engine = window.preview.engine
            engine.frameReady.connect(lambda *args: frames.append(args))
            engine.failed.connect(failures.append)
            try:
                window.input_path.setText(path)
                window.output_path.setText(tmp)
                self.assertTrue(wait_until(lambda: frames or failures, 20), 'initial preview did not finish')
                self.assertFalse(failures, failures)
                self.assertEqual(image_hash(frames[-1][0]), image_hash(frames[-1][1]))
                window.creative_panel.apply_preset('Creative 600 nits')
                window.creative_panel.controls['temperature'].setValue(.5)
                original_cache = dict(engine.cache)
                count = len(frames)
                self.assertTrue(wait_until(lambda: len(frames) > count or failures, 20))
                self.assertFalse(failures, failures)
                self.assertNotEqual(image_hash(frames[-1][0]), image_hash(frames[-1][1]))
                self.assertEqual(dict(engine.cache), original_cache)
                window.preview.position.setValue(.5)
                count = len(frames)
                self.assertTrue(wait_until(lambda: len(frames) > count or failures, 20))
                self.assertFalse(failures, failures)
                self.assertGreaterEqual(frames[-1][2], .49)
                directory = os.environ.get('HDR_SCREENSHOT_DIR')
                if directory:
                    Path(directory).mkdir(parents=True, exist_ok=True)
                    self.assertTrue(window.grab().save(str(Path(directory) / 'workspace.png')))
                    window.resize(1000, 680)
                    QtTest.QTest.qWait(100)
                    self.assertTrue(window.grab().save(str(Path(directory) / 'workspace-small.png')))
            finally:
                window.close()
                self.assertTrue(engine.closed)
                self.assertIsNone(engine.process)


if __name__ == '__main__':
    unittest.main()
