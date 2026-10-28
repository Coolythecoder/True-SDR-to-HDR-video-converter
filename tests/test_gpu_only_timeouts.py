"""Movie-length subprocesses must remain cancellable without a fixed deadline."""
from fractions import Fraction
import subprocess
import sys
import unittest
from unittest import mock

import gpu_only_export as gpu
from youtube_export_analysis import ExportCancelled


class MovieTimeoutTests(unittest.TestCase):
    def test_full_movie_command_can_finish_after_hours_with_elapsed_log(self):
        process, register, log = mock.Mock(), mock.Mock(), mock.Mock()
        process.communicate.side_effect = [subprocess.TimeoutExpired('probe', .2),
                                           (b'complete', b''), (b'complete', b'')]
        process.returncode = process.poll.return_value = 0
        with mock.patch.object(gpu.subprocess, 'Popen', return_value=process), \
             mock.patch.object(gpu.time, 'monotonic', side_effect=[0, 0, 7201]):
            result = gpu._run(['probe'], lambda: False, register, timeout=None,
                              stage='input timestamp scan', log=log)
        self.assertEqual(result, b'complete')
        self.assertIn('7201s elapsed', log.call_args.args[0])
        self.assertIn('input timestamp scan still running', log.call_args.args[0])
        process.kill.assert_not_called()
        self.assertEqual(register.call_args_list, [mock.call(process), mock.call(None)])

    def test_small_probe_timeout_names_stage_and_cleans_up_process(self):
        process, register = mock.Mock(), mock.Mock()
        process.poll.return_value = None
        with mock.patch.object(gpu.subprocess, 'Popen', return_value=process), \
             mock.patch.object(gpu.time, 'monotonic', side_effect=[0, 121]):
            with self.assertRaisesRegex(ValueError, 'startup colour-reference check timed out after 120 seconds'):
                gpu._run(['probe'], lambda: False, register, stage='startup colour-reference check')
        process.kill.assert_called_once()
        process.communicate.assert_called_once_with()
        register.assert_called_with(None)

    def test_cancel_terminates_and_reaps_real_unlimited_subprocess(self):
        registered = []
        checks = 0

        def cancelled():
            nonlocal checks
            checks += 1
            return checks >= 4

        with self.assertRaises(ExportCancelled):
            gpu._run([sys.executable, '-c', 'import time; time.sleep(60)'],
                     cancelled, registered.append, timeout=None, stage='MP4 packaging')
        self.assertIsNotNone(registered[0].poll())
        self.assertIsNone(registered[-1])

    def test_already_cancelled_does_not_start_command(self):
        with mock.patch.object(gpu.subprocess, 'Popen') as popen:
            with self.assertRaises(ExportCancelled):
                gpu._run(['probe'], lambda: True, lambda p: None, timeout=None)
        popen.assert_not_called()

    def test_command_error_preserves_diagnostic_and_stage(self):
        process = mock.Mock()
        process.returncode = process.poll.return_value = 1
        process.communicate.return_value = (b'', b'No space left on device')
        with mock.patch.object(gpu.subprocess, 'Popen', return_value=process):
            with self.assertRaisesRegex(ValueError, 'MP4 packaging failed: No space left on device'):
                gpu._run(['mux'], lambda: False, lambda p: None, timeout=None, stage='MP4 packaging')

    def test_output_movie_count_has_no_deadline_but_startup_can_be_bounded(self):
        info = gpu.SourceInfo(256,256,8,Fraction(24),Fraction(1,24000),False,False)
        for kwargs, expected in (({}, None), ({'timeout': 120}, 120)):
            with self.subTest(kwargs=kwargs), mock.patch.object(gpu, '_run', side_effect=ExportCancelled) as run:
                with self.assertRaises(ExportCancelled):
                    gpu.verify_output('probe', 'movie.mp4', info, 100000, 200, 50,
                                      lambda: False, lambda p: None, **kwargs)
                self.assertEqual(run.call_args.args[3], expected)
                self.assertEqual(run.call_args.kwargs['stage'], 'output frame-count verification')
