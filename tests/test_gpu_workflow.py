"""GPU integration contracts and actual asynchronous software-OpenCL preview."""
from contextlib import ExitStack
import io
import os
from pathlib import Path
import shutil
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PyQt5 import QtWidgets, QtTest
import SDR_to_HDR_cross_platform as app_module
from creative_hdr import CreativeHDRSettings
from gpu_colour import GPUPlan
from preview_widget import PreviewEngine


class GPUWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.window = app_module.SDRtoHDRConverter()
        self.directory = tempfile.TemporaryDirectory()
        self.source = Path(self.directory.name)/'clip.mp4'
        self.source.write_bytes(b'test source')
        self.window.input_path.setText(str(self.source))
        self.window.output_path.setText(self.directory.name)

    def tearDown(self):
        worker = self.window.conversion_thread
        if worker:
            for resource in worker.resources:
                resource.close()
            worker.resources.clear()
        self.window.close()
        self.directory.cleanup()
        self.app.processEvents()

    def start(self, youtube=True):
        with ExitStack() as stack:
            for key,value in {
                'find_ffmpeg':'ffmpeg','find_ffprobe':'ffprobe','has_zscale':True,
                'supports_libx265':True,'require_creative_filters':None,
                'probe_sdr_profile':app_module.DEFAULT_SDR_PROFILE,
                'validate_youtube_source':{'width':1280,'height':720,'has_audio':False},
                'probe_grade_geometry':{'width':1280,'height':720},
            }.items():
                stack.enter_context(mock.patch.object(app_module,key,return_value=value))
            stack.enter_context(mock.patch.object(app_module.ConversionThread,'start'))
            self.window._start_conversion(youtube_mode=youtube)
        self.assertIsNotNone(self.window.conversion_thread,self.window.status_label.text())
        return self.window.conversion_thread

    def test_defaults_are_separate_from_nvenc_and_youtube_quality(self):
        w = self.window
        self.assertTrue(w.processing_gpu.isChecked())
        self.assertTrue(w.hardware_decode.isChecked())
        self.assertFalse(w.gpu_checkbox.isChecked())
        self.assertEqual(w.youtube_quality.currentText(),'High')
        w.processing_gpu.setChecked(False)
        self.assertFalse(w.preview.engine.acceleration.enabled)
        self.assertEqual(w.youtube_quality.currentText(),'High')
        self.assertFalse(w.gpu_checkbox.isChecked())

    def test_accelerated_snapshot_keeps_x265_settings_and_graded_metadata(self):
        w = self.window
        w.creative_panel.setChecked(True)
        w.youtube_quality.setCurrentText('Maximum')
        worker = self.start()
        original = worker.ffmpeg_commands[0]
        plan = GPUPlan(CreativeHDRSettings(enabled=True))
        w.processing_gpu.setChecked(False)
        w.hardware_decode.setChecked(False)
        w.creative_panel.apply_preset('Bright 1000 nits')
        with mock.patch.object(app_module,'choose_gpu_plan',return_value=plan) as choose, \
             mock.patch.object(app_module,'choose_nvidia_decoder',return_value=None) as decoder, \
             mock.patch.object(app_module,'measure_content_light',return_value=(400,100,3)) as measure:
            prepared = worker.prepare_job(original,worker)
        choose.assert_called_once()
        decoder.assert_called_once()
        self.assertEqual(choose.call_args.args[1].peak_nits,600)
        self.assertIs(measure.call_args.kwargs['gpu_plan'],plan)
        self.assertIn('program_opencl',prepared[prepared.index('-vf')+1])
        self.assertEqual(prepared[prepared.index('-c:v')+1],'libx265')
        self.assertEqual(prepared[prepared.index('-crf')+1],'12')
        self.assertEqual(prepared[prepared.index('-preset')+1],'slow')
        self.assertEqual(prepared[prepared.index('-b:a')+1],'384k')
        params = prepared[prepared.index('-x265-params')+1]
        self.assertIn('L(6000000,0)',params)
        self.assertIn('max-cll=400,100',params)

    def test_failed_gpu_scan_restarts_entire_cpu_scan_and_uses_cpu_encode(self):
        self.window.creative_panel.setChecked(True)
        worker = self.start()
        original = worker.ffmpeg_commands[0]
        plan = GPUPlan(CreativeHDRSettings(enabled=True))
        logs = []
        worker.log_output.connect(logs.append)
        with mock.patch.object(app_module,'choose_gpu_plan',return_value=plan), \
             mock.patch.object(app_module,'choose_nvidia_decoder',return_value=None), \
             mock.patch.object(app_module,'measure_content_light',side_effect=[ValueError('device lost'),(399,90,3)]) as scan:
            result = worker.prepare_job(original,worker)
        self.assertEqual(scan.call_count,2)
        self.assertIs(scan.call_args_list[0].kwargs['gpu_plan'],plan)
        self.assertNotIn('gpu_plan',scan.call_args_list[1].kwargs)
        self.assertEqual(result[result.index('-vf')+1],original[original.index('-vf')+1])
        self.assertNotIn('-init_hw_device',result)
        self.assertIn('max-cll=399,90',result[result.index('-x265-params')+1])
        self.assertTrue(any('restarting the complete scan' in text for text in logs))

    def test_unavailable_gpu_keeps_manual_metadata_and_encoder(self):
        self.window.override_metadata.setChecked(True)
        self.window.max_cll_input.setText('180')
        self.window.max_fall_input.setText('90')
        worker = self.start()
        original = worker.ffmpeg_commands[0]
        with mock.patch.object(app_module,'choose_nvidia_decoder',return_value=None), \
             mock.patch.object(app_module,'choose_gpu_plan') as colour, \
             mock.patch.object(app_module,'measure_content_light') as scan:
            prepared = worker.prepare_job(original,worker)
        scan.assert_not_called()
        colour.assert_not_called()
        self.assertEqual(prepared,original)

    def test_custom_creative_uses_gpu_without_changing_custom_quality(self):
        self.window.creative_panel.setChecked(True)
        self.window.cq_slider.setValue(19)
        worker = self.start(youtube=False)
        plan = GPUPlan(CreativeHDRSettings(enabled=True))
        with mock.patch.object(app_module,'choose_gpu_plan',return_value=plan), \
             mock.patch.object(app_module,'choose_nvidia_decoder',return_value=None), \
             mock.patch.object(app_module,'measure_content_light',return_value=(400,110,3)):
            result = worker.prepare_job(worker.ffmpeg_commands[0],worker)
        self.assertEqual(result[result.index('-crf')+1],'19')
        self.assertIn('program_opencl',result[result.index('-vf')+1])

    def test_shader_resources_closed_on_worker_failure_and_cancellation(self):
        for error in (ValueError('scan failed'),app_module.ExportCancelled()):
            with self.subTest(error=type(error).__name__):
                worker = app_module.ConversionThread([['ffmpeg','-i','input','out']],
                         prepare_job=mock.Mock(side_effect=error))
                resource = mock.Mock()
                worker.resources.append(resource)
                worker.run()
                resource.close.assert_called_once()
                self.assertEqual(worker.resources,[])


class GPUPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.engine = PreviewEngine()
        self.engine.ffmpeg = 'ffmpeg'
        self.engine.source = SimpleNamespace(path='sample',fps=30,duration=1,proxy_size=lambda bound:(16,16))
        self.grade = CreativeHDRSettings(enabled=True)

    def tearDown(self):
        self.engine.shutdown()
        self.engine.deleteLater()
        self.app.processEvents()

    def launch_fake(self):
        callback = mock.Mock()
        patch = mock.patch.object(self.engine,'_launch')
        launch = patch.start()
        self.addCleanup(patch.stop)
        self.engine.acceleration.render(16,16,self.grade,200,callback,b'',2048)
        return callback,launch,launch.call_args

    def test_preview_device_error_falls_back_and_cleans_shader(self):
        callback,launch,call = self.launch_fake()
        call.kwargs['cleanup']()
        call.kwargs['on_error']('no OpenCL')
        self.assertEqual(self.engine.acceleration.device_index,1)
        self.assertEqual(len(self.engine.acceleration.plans),0)
        callback.assert_not_called()
        self.assertTrue(self.engine.debounce.isActive())

    def test_incorrect_gpu_preview_uses_cpu_reference(self):
        callback,launch,call = self.launch_fake()
        call.kwargs['cleanup']()
        call.args[2](bytes([150])*(16*16*6))
        reference = bytes([80])*(16*16*6)
        launch.call_args.args[2](reference)
        callback.assert_called_once_with(reference)
        self.assertIn('parity check failed',self.engine.acceleration.backend)
        self.assertFalse(self.engine.acceleration.verified)

    def test_stale_gpu_preview_never_replaces_newer_grade(self):
        callback,launch,call = self.launch_fake()
        self.engine.revision += 1
        call.kwargs['cleanup']()
        call.args[2](bytes([80])*(16*16*6))
        callback.assert_not_called()
        self.assertEqual(launch.call_count,1)

    @unittest.skipUnless(os.environ.get('HDR_TEST_OPENCL_DEVICE') and shutil.which('ffmpeg'),
                         'Requires software OpenCL test environment')
    def test_actual_qprocess_gpu_preview_and_cleanup(self):
        self.engine.ffmpeg = shutil.which('ffmpeg')
        self.engine.grade = self.grade
        self.engine.cache[self.engine._key()] = np.ones((3,16,16),dtype='<f4').tobytes()
        frames, errors = [], []
        self.engine.frameReady.connect(lambda *values: frames.append(values))
        self.engine.failed.connect(errors.append)
        with mock.patch('gpu_preview.GPU_DEVICES',(os.environ['HDR_TEST_OPENCL_DEVICE'],)):
            self.engine._schedule()
            deadline = time.monotonic()+60
            while not frames and not errors and time.monotonic()<deadline:
                QtTest.QTest.qWait(20)
        self.assertFalse(errors,errors)
        self.assertTrue(frames)
        self.assertTrue(self.engine.acceleration.verified)
        self.assertEqual(self.engine.acceleration.backend,'OpenCL GPU grading')
        self.assertEqual(len(self.engine.acceleration.plans),0)


if __name__ == '__main__':
    unittest.main()
