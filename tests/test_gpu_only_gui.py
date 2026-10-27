"""Optional-mode isolation, frozen worker settings, and fail-closed behaviour."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from PyQt5 import QtWidgets
import gpu_only_studio as studio
from creative_hdr import CreativeHDRSettings
from gpu_only_export import GPUOnlySettings
from youtube_export_analysis import ExportCancelled


class OptionalModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.window = studio.GPUOnlyStudio()

    def tearDown(self):
        self.window.conversion_thread = None
        self.window.close()
        self.app.processEvents()

    def test_default_off_and_toggle_restores_normal_controls(self):
        w = self.window
        self.assertFalse(w.gpu_only_checkbox.isChecked())
        w.gpu_checkbox.setChecked(False)
        w.bit_depth.setCurrentText('8')
        w.gpu_only_checkbox.setChecked(True)
        self.assertTrue(w.gpu_checkbox.isChecked())
        self.assertEqual(w.bit_depth.currentText(), '10')
        self.assertFalse(w.gpu_checkbox.isEnabled())
        w.gpu_only_checkbox.setChecked(False)
        self.assertFalse(w.gpu_checkbox.isChecked())
        self.assertEqual(w.bit_depth.currentText(), '8')
        self.assertTrue(w.gpu_checkbox.isEnabled())

    def test_youtube_always_delegates_to_original_builder(self):
        w = self.window
        for enabled in (False, True):
            w.gpu_only_checkbox.setChecked(enabled)
            with mock.patch.object(studio.SDRtoHDRConverter, '_start_conversion') as original, \
                 mock.patch.object(studio, 'GPUOnlyThread') as native:
                w._start_conversion(youtube_mode=True)
                original.assert_called_once_with(youtube_mode=True)
                native.assert_not_called()

    def test_disabled_custom_uses_original_builder(self):
        with mock.patch.object(studio.SDRtoHDRConverter, '_start_conversion') as original:
            self.window._start_conversion()
            original.assert_called_once_with(youtube_mode=False)

    def test_native_worker_receives_frozen_grade_and_separate_filename(self):
        w = self.window
        with tempfile.TemporaryDirectory() as td:
            source = Path(td)/'test.mp4'
            source.write_bytes(b'test')
            w.input_path.setText(str(source))
            w.output_path.setText(td)
            w.creative_panel.setChecked(True)
            w.gpu_only_checkbox.setChecked(True)
            w.cq_slider.setValue(19)
            with mock.patch.object(studio,'find_ffmpeg',return_value='ffmpeg'), \
                 mock.patch.object(studio,'find_ffprobe',return_value='ffprobe'), \
                 mock.patch.object(studio.GPUOnlyThread,'start'):
                w._start_conversion()
            worker = w.conversion_thread
            self.assertIsInstance(worker, studio.GPUOnlyThread, w.status_label.text())
            self.assertTrue(worker.jobs[0][1].endswith('_HDR_GPU.mp4'))
            self.assertEqual(worker.settings.qp,19)
            w.creative_panel.apply_preset('Bright 1000 nits')
            self.assertEqual(worker.settings.grade.peak_nits,600)
            self.assertFalse(w.export_page.isEnabled())

    def test_missing_dependencies_stop_without_original_export_or_fallback(self):
        worker = studio.GPUOnlyThread([('in.mp4','out.mp4')],'ffmpeg','ffprobe',GPUOnlySettings())
        logs=[]; statuses=[]
        worker.log_output.connect(logs.append)
        worker.update_status.connect(statuses.append)
        with mock.patch.object(studio,'export_gpu_only',side_effect=ValueError('Missing CUDA')) as export, \
             mock.patch.object(studio.ConversionThread,'run') as old:
            worker.run()
            export.assert_called_once()
            old.assert_not_called()
        self.assertIn('1 failed', statuses[-1])
        self.assertTrue(any('CPU fallback is disabled' in line for line in logs))

    def test_cancellation_is_not_reported_as_success(self):
        worker=studio.GPUOnlyThread([('in','out')],'ffmpeg','ffprobe',GPUOnlySettings())
        statuses=[];worker.update_status.connect(statuses.append)
        with mock.patch.object(studio,'export_gpu_only',side_effect=ExportCancelled):
            worker.run()
        self.assertIn('Canceled',statuses[-1])
        self.assertIn('0 succeeded',statuses[-1])


if __name__ == '__main__':
    unittest.main()
