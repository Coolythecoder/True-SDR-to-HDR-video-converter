"""Regression tests for a partially black GPU preview after initial validation."""
from dataclasses import replace
import os
import shutil
import subprocess
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from creative_hdr import CreativeHDRSettings
from gpu_preview import PreviewAcceleration
from grading_preview import render_command


class PreviewRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.engine = SimpleNamespace(
            closed=False, paused=False, source=object(), revision=0,
            ffmpeg='ffmpeg', debounce=mock.Mock(), message=mock.Mock(),
            failed=mock.Mock(), _key=mock.Mock(return_value=('clip', 0, 640)),
            _launch=mock.Mock())
        self.acc = PreviewAcceleration(self.engine)
        self.grade = CreativeHDRSettings(enabled=True)
        self.plan_patch = mock.patch('gpu_preview.GPUPlan')
        self.plan = self.plan_patch.start().return_value
        self.command_patch = mock.patch('gpu_preview.render_command', return_value=['ffmpeg'])
        self.command_patch.start()
        self.addCleanup(self.command_patch.stop)
        self.addCleanup(self.plan_patch.stop)
        self.addCleanup(self.acc.close)

    def begin(self, gpu_bytes, grade=None, width=16, height=16):
        callback = mock.Mock()
        self.engine._launch.reset_mock()
        self.assertTrue(self.acc.render(width, height, grade or self.grade, 200,
                                       callback, b'cached source', width*height*6+1024))
        gpu_call = self.engine._launch.call_args
        gpu_call.kwargs['cleanup']()
        gpu_call.args[2](gpu_bytes)
        # No unchecked GPU result may be published, even after a valid first frame.
        callback.assert_not_called()
        self.assertEqual(self.engine._launch.call_count, 2)
        return callback, self.engine._launch.call_args

    def validate(self, gpu_bytes, cpu_bytes, **kwargs):
        callback, reference_call = self.begin(gpu_bytes, **kwargs)
        reference_call.args[2](cpu_bytes)
        return callback

    def prime(self, value=80):
        data = bytes([value]) * (16*16*6)
        callback = self.validate(data, data)
        callback.assert_called_once_with(data)
        self.assertTrue(self.acc.verified)

    def test_black_opening_frame_cannot_authorize_corrupt_later_frame(self):
        self.prime(0)
        self.engine.revision += 1
        self.engine._key.return_value = ('clip', 120, 640)
        correct = np.full((16, 32, 3), 96, dtype=np.uint8)
        corrupt = correct.copy()
        # Second half is the graded image; its right half becomes black.
        corrupt[:, 24:, :] = 0
        callback = self.validate(corrupt.tobytes(), correct.tobytes())
        callback.assert_called_once_with(correct.tobytes())
        self.assertFalse(self.acc.verified)
        self.assertIn('parity check failed', self.acc.backend)

    def test_changed_grade_is_checked_again_on_the_same_frame(self):
        self.prime()
        self.engine.revision += 1
        correct = bytes([110]) * (16*16*6)
        callback = self.validate(bytes(len(correct)), correct,
                                 grade=replace(self.grade, saturation=1.3))
        callback.assert_called_once_with(correct)
        self.assertFalse(self.acc.verified)

    def test_changed_resolution_is_checked_again(self):
        self.prime()
        self.engine.revision += 1
        self.engine._key.return_value = ('clip', 0, 960)
        correct = bytes([90]) * (32*18*6)
        callback = self.validate(bytes(len(correct)), correct, width=32, height=18)
        callback.assert_called_once_with(correct)

    def test_new_source_cannot_reuse_previous_validation(self):
        self.prime()
        self.engine.source = object()
        self.engine.revision += 1
        self.engine._key.return_value = ('other clip', 0, 640)
        correct = bytes([100]) * (16*16*6)
        callback = self.validate(bytes(len(correct)), correct)
        callback.assert_called_once_with(correct)

    def test_failure_stays_disabled_and_reason_remains_visible(self):
        correct = bytes([100]) * (16*16*6)
        self.validate(bytes(len(correct)), correct)
        self.engine._launch.reset_mock()
        self.assertFalse(self.acc.render(16, 16, self.grade, 200, mock.Mock(), b'', 2048))
        self.engine._launch.assert_not_called()
        self.assertIn('parity check failed', self.acc.backend)

    def test_stale_reference_never_publishes_or_quarantines_device(self):
        data = bytes([80]) * (16*16*6)
        callback, call = self.begin(data)
        self.engine.revision += 1
        call.args[2](bytes(len(data)))
        callback.assert_not_called()
        self.assertEqual(self.acc.device_index, 0)
        self.assertFalse(self.acc.verified)

    def test_reset_invalidates_pending_reference_even_without_revision_change(self):
        data = bytes([80]) * (16*16*6)
        callback, call = self.begin(data)
        self.acc.reset(False)
        call.args[2](data)
        callback.assert_not_called()
        self.assertFalse(self.acc.verified)

    def test_reference_error_never_publishes_unverified_gpu_pixels(self):
        data = bytes([80]) * (16*16*6)
        callback, call = self.begin(data)
        self.assertIn('on_error', call.kwargs)
        call.kwargs['on_error']('CPU reference unavailable')
        callback.assert_not_called()
        self.engine.failed.emit.assert_called_once()
        self.assertFalse(self.acc.verified)

    def test_incomplete_reference_is_an_error_not_a_repeated_gpu_retry(self):
        data = bytes([80]) * (16*16*6)
        callback, call = self.begin(data)
        call.args[2](b'incomplete')
        callback.assert_not_called()
        self.engine.failed.emit.assert_called_once()
        self.assertFalse(self.engine.debounce.start.called)

    def test_close_and_pause_do_not_restart_validation(self):
        for attribute in ('closed', 'paused'):
            with self.subTest(attribute=attribute):
                self.acc.reset(True)
                setattr(self.engine, attribute, False)
                data = bytes([80]) * (16*16*6)
                callback, call = self.begin(data)
                self.engine.debounce.reset_mock()
                setattr(self.engine, attribute, True)
                call.args[2](data)
                callback.assert_not_called()
                self.engine.debounce.start.assert_not_called()
                setattr(self.engine, attribute, False)

    def test_matching_later_frame_is_validated_and_published(self):
        self.prime()
        self.engine.revision += 1
        data = bytes([160]) * (16*16*6)
        callback = self.validate(data, data)
        callback.assert_called_once_with(data)
        self.assertEqual(self.acc.backend, 'OpenCL GPU grading')

    def test_reset_explicitly_allows_retry_after_rejection(self):
        correct = bytes([100]) * (16*16*6)
        self.validate(bytes(len(correct)), correct)
        self.acc.reset(True)
        callback = self.validate(correct, correct)
        callback.assert_called_once_with(correct)
        self.assertTrue(self.acc.verified)


@unittest.skipUnless(os.environ.get('HDR_TEST_OPENCL_DEVICE') and shutil.which('ffmpeg'),
                     'Requires explicit software OpenCL test environment')
class FullSizePreviewTests(unittest.TestCase):
    def test_actual_preview_sizes_and_right_edge_match_cpu(self):
        from PyQt5 import QtWidgets, QtTest
        from preview_widget import PreviewEngine
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        engine = PreviewEngine()
        engine.ffmpeg = shutil.which('ffmpeg')
        grade = CreativeHDRSettings(enabled=True, temperature=.2, saturation=1.1)
        errors = []
        engine.failed.connect(errors.append)
        try:
            with mock.patch('gpu_preview.GPU_DEVICES', (os.environ['HDR_TEST_OPENCL_DEVICE'],)):
                for width, height in ((480, 270), (640, 360), (960, 540)):
                    with self.subTest(width=width, height=height):
                        engine.revision += 1
                        engine.source = SimpleNamespace(
                            path='pattern', fps=30, duration=1,
                            proxy_size=lambda bound, w=width, h=height: (w, h))
                        engine.bound = width
                        yy, xx = np.mgrid[:height, :width]
                        pixels = np.stack((
                            .05 + .8*xx/max(1, width-1),
                            .08 + .6*yy/max(1, height-1),
                            .1 + .7*((xx//23 + yy//19) % 2))).astype('<f4')
                        payload = pixels.tobytes()
                        reference = subprocess.run(
                            render_command(engine.ffmpeg, width, height, grade, 200),
                            input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=60, check=True).stdout
                        frames = []
                        accepted = engine.acceleration.render(
                            width, height, grade, 200, frames.append, payload, width*height*6+1024)
                        self.assertTrue(accepted)
                        deadline = time.monotonic()+90
                        while not frames and not errors and time.monotonic() < deadline:
                            QtTest.QTest.qWait(20)
                        self.assertFalse(errors, errors)
                        self.assertEqual(len(frames), 1)
                        actual = np.frombuffer(frames[0], dtype=np.uint8).reshape(height, width*2, 3)
                        expected = np.frombuffer(reference, dtype=np.uint8).reshape(height, width*2, 3)
                        difference = np.abs(actual.astype(np.int16)-expected.astype(np.int16))
                        self.assertLessEqual(difference.max(), 2)
                        self.assertLessEqual(difference.mean(), .15)
                        self.assertGreater(float(actual[:, -16:, :].mean()), 10)
                        self.assertEqual(engine.acceleration.backend, 'OpenCL GPU grading')
                        self.assertTrue(engine.acceleration.verified)
        finally:
            engine.shutdown()
            engine.deleteLater()
            app.processEvents()


if __name__ == '__main__':
    unittest.main()
