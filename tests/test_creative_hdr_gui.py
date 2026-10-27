"""GUI contracts for optional HDR grading without starting a real conversion."""
from contextlib import ExitStack
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
try:
    from PyQt5 import QtWidgets
    import SDR_to_HDR_cross_platform as app_module
except ImportError:
    QtWidgets = None
    app_module = None


@unittest.skipIf(app_module is None, 'Requires GUI dependencies')
class CreativeGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.widget = app_module.SDRtoHDRConverter()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.source = Path(self.directory.name)/'video.mp4'
        self.source.write_bytes(b'placeholder')
        self.widget.input_path.setText(str(self.source))
        self.widget.output_path.setText(self.directory.name)

    def tearDown(self):
        self.widget.close()

    def start_mocked(self, youtube=True, probe='ffprobe'):
        with ExitStack() as stack:
            for function, result in {
                'find_ffmpeg': 'ffmpeg', 'find_ffprobe': probe,
                'has_zscale': True, 'supports_libx265': True,
                'require_creative_filters': None,
                'probe_sdr_profile': app_module.DEFAULT_SDR_PROFILE,
                'validate_youtube_source': {'width': 1280, 'height': 720, 'has_audio': False},
                'probe_grade_geometry': {'width': 1280, 'height': 720},
            }.items():
                stack.enter_context(mock.patch.object(app_module, function, return_value=result))
            stack.enter_context(mock.patch.object(app_module.ConversionThread, 'start'))
            self.widget._start_conversion(youtube_mode=youtube)
        return self.widget.conversion_thread

    def test_accurate_is_default_and_reset_disables_grade(self):
        panel = self.widget.creative_panel
        self.assertFalse(panel.settings().enabled)
        panel.apply_preset('Bright 1000 nits')
        self.assertEqual(panel.settings().peak_nits, 1000)
        self.assertTrue(panel.settings().enabled)
        panel.reset_neutral()
        self.assertFalse(panel.settings().enabled)
        command = self.widget.build_ffmpeg_command('ffmpeg', 'in.mp4', 'out.mp4', False)
        self.assertNotIn('geq=', command[command.index('-vf')+1])

    def test_normal_and_youtube_include_grade(self):
        self.widget.creative_panel.setChecked(True)
        command = self.widget.build_ffmpeg_command('ffmpeg', 'in.mp4', 'out.mp4', False)
        self.assertIn('geq=', command[command.index('-vf')+1])
        worker = self.start_mocked()
        self.assertIsNotNone(worker)
        command = worker.ffmpeg_commands[0]
        self.assertIn('geq=', command[command.index('-vf')+1])
        self.assertIsNotNone(worker.prepare_job)
        self.assertIn('L(6000000,0)', command[command.index('-x265-params')+1])

    def test_grade_forces_post_grade_analysis_and_ignores_stale_manual_metadata(self):
        self.widget.youtube_analyse_light.setChecked(False)
        self.widget.override_metadata.setChecked(True)
        self.widget.max_cll_input.setText('not a number')
        self.widget.creative_panel.setChecked(True)
        self.assertFalse(self.widget.max_cll_input.isEnabled())
        worker = self.start_mocked()
        self.assertIsNotNone(worker)
        command = worker.ffmpeg_commands[0]
        self.assertIn('max-cll=0,0', command[command.index('-x265-params')+1])
        self.widget.creative_panel.apply_preset('Bright 1000 nits')
        with mock.patch.object(app_module, 'measure_content_light', return_value=(450, 110, 3)) as measure:
            prepared = worker.prepare_job(command, worker)
        self.assertEqual(measure.call_args.kwargs['grade'].peak_nits, 600)
        params = prepared[prepared.index('-x265-params')+1]
        self.assertIn('L(6000000,0)', params)
        self.assertIn('max-cll=450,110', params)
        self.assertEqual(prepared[prepared.index('-vf')+1], command[command.index('-vf')+1])

    def test_normal_creative_also_scans_after_grade(self):
        self.widget.creative_panel.setChecked(True)
        worker = self.start_mocked(youtube=False)
        self.assertIsNotNone(worker)
        self.assertIsNotNone(worker.prepare_job)
        with mock.patch.object(app_module, 'measure_content_light', return_value=(400, 150, 3)):
            command = worker.prepare_job(worker.ffmpeg_commands[0], worker)
        self.assertIn('max-cll=400,150', command[command.index('-x265-params')+1])

    def test_eight_bit_standard_output_is_rejected_not_mistagged(self):
        self.widget.bit_depth.setCurrentText('8')
        self.widget.creative_panel.setChecked(True)
        worker = self.start_mocked(youtube=False)
        self.assertIsNone(worker)
        self.assertIn('10-bit', self.widget.status_label.text())

    def test_missing_ffprobe_fails_before_worker(self):
        self.widget.creative_panel.setChecked(True)
        self.assertIsNone(self.start_mocked(probe=None))
        self.assertIn('FFprobe', self.widget.status_label.text())

    def test_peak_below_white_rejected(self):
        self.widget.hdr_reference_nits.setValue(400)
        self.widget.creative_panel.setChecked(True)
        self.widget.creative_panel.controls['peak_nits'].setValue(200)
        self.assertIsNone(self.start_mocked())
        self.assertIn('below reference white', self.widget.status_label.text())


if __name__ == '__main__':
    unittest.main()
