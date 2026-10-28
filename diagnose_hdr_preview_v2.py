"""Read-only HDR GPU diagnostics with an explicit current-frame test button.

Save beside SDR_to_HDR_cross_platform.py and run:
    python diagnose_hdr_preview_v2.py

The GUI runs a synthetic hardware self-test automatically, then allows an
explicit test of the cached current frame (even when the app uses CPU fallback).
No application source, grading controls, export settings or video files are
modified. No data is uploaded. Reports contain paths, settings, code hashes
and numerical comparisons, not images/audio. A .json report is published only
when the checks have finished; intermediate data has an .inprogress suffix.

For a hardware-only test without opening the GUI:
    python diagnose_hdr_preview_v2.py --headless
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time


def compare_bytes(reference, actual, shape, floating=False):
    import numpy as np
    dtype = '<f4' if floating else 'u1'
    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    result = {'expected_bytes': expected, 'reference_bytes': len(reference),
              'actual_bytes': len(actual)}
    if len(reference) != expected or len(actual) != expected:
        result['error'] = 'Incomplete frame'
        return result
    a = np.frombuffer(reference, dtype=dtype).reshape(shape).astype(np.float64)
    b = np.frombuffer(actual, dtype=dtype).reshape(shape).astype(np.float64)
    result['reference_nonfinite'] = int((~np.isfinite(a)).sum())
    result['actual_nonfinite'] = int((~np.isfinite(b)).sum())
    result['bit_exact'] = reference == actual
    if result['reference_nonfinite'] or result['actual_nonfinite']:
        result['error'] = 'Non-finite samples'
        return result
    diff = np.abs(a - b)
    index = tuple(int(x) for x in np.unravel_index(diff.argmax(), shape))
    result.update(max_difference=float(diff.max()), mean_difference=float(diff.mean()),
                  p99_difference=float(np.quantile(diff, .99)),
                  differing_samples=int(np.count_nonzero(diff)),
                  worst_index=index, reference_at_worst=float(a[index]),
                  actual_at_worst=float(b[index]))
    if floating:
        result['matches_existing_float_tolerance'] = bool(np.allclose(a, b, rtol=3e-4, atol=3e-5))
    else:
        result['matches_existing_preview_tolerance'] = bool(diff.max() <= 2 and diff.mean() <= .15)
        result['samples_over_2'] = int((diff > 2).sum())
    return result


def preview_comparison(reference, actual, width, height):
    import numpy as np
    result = {'whole_pair': compare_bytes(reference, actual, (height, width * 2, 3))}
    if 'error' in result['whole_pair']:
        return result
    a = np.frombuffer(reference, dtype='u1').reshape(height, width * 2, 3)
    b = np.frombuffer(actual, dtype='u1').reshape(height, width * 2, 3)
    for name, left, right in [('before', 0, width), ('after', width, width * 2),
                              ('after_left_half', width, width + width // 2),
                              ('after_right_half', width + width // 2, width * 2)]:
        result[name] = compare_bytes(a[:, left:right].tobytes(), b[:, left:right].tobytes(),
                                     (height, right - left, 3))
    return result


def cpu_command(ffmpeg, width, height, graph, pixel_format='gbrpf32le'):
    return [ffmpeg, '-hide_banner', '-nostdin', '-v', 'error', '-f', 'rawvideo',
            '-pixel_format', 'gbrpf32le', '-video_size', '{}x{}'.format(width, height),
            '-framerate', '1', '-i', 'pipe:0', '-vf', graph, '-frames:v', '1',
            '-pix_fmt', pixel_format, '-f', 'rawvideo', 'pipe:1']


def save_report(path, data):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)



class DiagnosticCancelled(Exception):
    pass


def run_tool(command, payload=None, cancelled=None, timeout=30):
    """Bounded subprocess; no shell, no GUI calls, kill and reap on cancellation."""
    if cancelled is not None and cancelled.is_set():
        raise DiagnosticCancelled()
    kwargs = dict(stdin=subprocess.PIPE if payload is not None else subprocess.DEVNULL,
                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if sys.platform == 'win32':
        kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
    process = subprocess.Popen(command, **kwargs)
    start, first = time.monotonic(), True
    try:
        while True:
            if cancelled is not None and cancelled.is_set():
                raise DiagnosticCancelled()
            try:
                output, errors = process.communicate(input=payload if first else None, timeout=.2)
                if process.returncode:
                    raise RuntimeError('Exit {}: {}'.format(process.returncode,
                        errors.decode('utf-8', errors='replace')[-6000:]))
                return output
            except subprocess.TimeoutExpired:
                first = False
                if time.monotonic() - start >= timeout:
                    raise RuntimeError('Diagnostic command timed out after {} seconds.'.format(timeout))
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate()
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                try:
                    stream.close()
                except OSError:
                    pass


def new_report(repo, scope):
    report = {'diagnostic_version': 2, 'scope': scope,
              'utc': datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'),
              'python': sys.version, 'python_executable': sys.executable,
              'source_folder': str(repo), 'complete': False, 'status': 'running',
              'module_sha256': {}, 'stages': {}}
    for name in ('gpu_colour.py', 'gpu_float_transport.py', 'gpu_preview.py',
                 'grading_preview.py', 'creative_hdr.py', 'preview_widget.py'):
        path = repo / name
        report['module_sha256'][name] = (hashlib.sha256(path.read_bytes()).hexdigest()
                                        if path.is_file() else 'Missing')
    # Fast optional provenance. No network or git changes.
    try:
        git = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                             capture_output=True, text=True, timeout=3)
        report['git_revision'] = git.stdout.strip() if git.returncode == 0 else 'Unavailable'
    except (OSError, subprocess.TimeoutExpired):
        report['git_revision'] = 'Unavailable'
    return report


def snapshot_from_engine(engine):
    """Copy an immutable frame/settings snapshot on the GUI thread only."""
    if engine.source is None:
        raise ValueError('Open an SDR video and let its preview appear first.')
    if engine.paused:
        raise ValueError('Finish or cancel the export before running diagnostics.')
    key = engine._key()
    payload = engine.cache.get(key)
    width, height = engine.source.proxy_size(engine.bound)
    if payload is None or len(payload) != width * height * 12:
        raise ValueError('The selected frame is not decoded yet. Click Refresh and let the image appear.')
    return {'scope': 'current_frame', 'ffmpeg': engine.ffmpeg,
            'width': width, 'height': height, 'payload': bytes(payload),
            'grade': asdict(engine.grade), 'white_nits': engine.white_nits,
            'source_path': engine.source.path, 'source_fps': engine.source.fps,
            'source_profile': dict(engine.source.profile),
            'preview_seconds': key[1] / engine.source.fps,
            'preview_backend_when_clicked': engine.acceleration.backend}


def synthetic_snapshot(ffmpeg):
    import numpy as np
    from creative_hdr import CreativeHDRSettings
    width, height = 640, 360
    y, x = np.mgrid[:height, :width]
    # Float GBR: dark/bright gradients, colour differences and HDR headroom.
    pixels = np.stack((.05 + x / width, .02 + y / height,
                       .1 + 1.6 * x / width * y / height)).astype('<f4')
    pixels[:, :8, :8] = 0
    return {'scope': 'hardware_selftest', 'ffmpeg': ffmpeg,
            'width': width, 'height': height, 'payload': pixels.tobytes(),
            'grade': asdict(CreativeHDRSettings(enabled=True)), 'white_nits': 200}


def investigate(snapshot, report, destination, events, cancelled, run=run_tool):
    """Always run explicitly; record independent stage failures instead of stalling."""
    destination = Path(destination)
    partial = destination.with_suffix('.inprogress')
    stages = report['stages']
    report['context'] = {k: v for k, v in snapshot.items() if k != 'payload'}
    report['context']['payload_sha256'] = hashlib.sha256(snapshot['payload']).hexdigest()
    report['context']['payload_bytes'] = len(snapshot['payload'])
    report['errors'] = []

    def checkpoint(name, value):
        stages[name] = value
        save_report(partial, report)
        events.put(('progress', name))

    def command(args, data=None):
        return run(args, data, cancelled=cancelled, timeout=30)

    def attempt(name, action):
        try:
            return action()
        except DiagnosticCancelled:
            raise
        except Exception as exc:
            report['errors'].append(name)
            checkpoint(name, {'error': '{}: {}'.format(type(exc).__name__, exc)})
            return None

    try:
        save_report(partial, report)
        ffmpeg = snapshot['ffmpeg']
        if not ffmpeg:
            raise RuntimeError('FFmpeg was not found on PATH. The application uses the same PATH discovery.')
        resolved = str(Path(shutil.which(ffmpeg) or ffmpeg).resolve())
        checkpoint('ffmpeg_path', resolved)
        version = attempt('ffmpeg_version', lambda: command([ffmpeg, '-version']))
        if version is not None:
            checkpoint('ffmpeg_version', version.decode('utf-8', errors='replace'))
        filters = attempt('ffmpeg_filters', lambda: command([ffmpeg, '-hide_banner', '-filters']))
        if filters is not None:
            names = {parts[1] for line in filters.decode('utf-8', errors='replace').splitlines()
                     for parts in [line.split()] if len(parts) >= 2}
            checkpoint('ffmpeg_filters', {name: name in names for name in
                       ('program_opencl', 'hwupload', 'hwdownload', 'mergeplanes',
                        'extractplanes', 'geq', 'zscale', 'tonemap', 'hstack')})
        from creative_hdr import CreativeHDRSettings, build_creative_filter
        from gpu_colour import GPUPlan, GPU_DEVICES
        from gpu_float_transport import merge_float_planes, LINEAR_BT2020_TAGS
        from grading_preview import viewing_filter, render_command
        width, height = snapshot['width'], snapshot['height']
        payload, white = snapshot['payload'], snapshot['white_nits']
        if len(payload) != width * height * 12:
            raise ValueError('Cached frame size does not match its dimensions.')
        grade = CreativeHDRSettings(**snapshot['grade'])
        if not grade.enabled:
            # Do not edit the GUI controls just to exercise the GPU shader.
            grade = CreativeHDRSettings(enabled=True)
            report['diagnostic_grade_note'] = 'Creative grade was disabled; a default creative grade is tested separately. App settings are unchanged.'
        report['tested_grade'] = asdict(grade)
        grade.validate(white)
        shape = (3, height, width)
        expected = attempt('cpu_grade', lambda: command(cpu_command(ffmpeg, width, height,
                        build_creative_filter(grade, white).lstrip(',') or 'null'), payload))
        if expected is not None:
            checkpoint('cpu_grade', {'bytes': len(expected), 'sha256': hashlib.sha256(expected).hexdigest()})
        extraction = ('split=3[diagr][diagg][diagb];[diagr]extractplanes=r[diagor];'
                      '[diagg]extractplanes=g[diagog];[diagb]extractplanes=b[diagob];')
        transport = attempt('cpu_float_transport', lambda: command(cpu_command(
                            ffmpeg, width, height, extraction + merge_float_planes('diag')), payload))
        if transport is not None:
            checkpoint('cpu_float_transport', compare_bytes(payload, transport, shape, True))
        cpu_pair = attempt('cpu_preview', lambda: command(render_command(
                           ffmpeg, width, height, grade, white), payload))
        if cpu_pair is not None:
            checkpoint('cpu_preview', {'bytes': len(cpu_pair), 'sha256': hashlib.sha256(cpu_pair).hexdigest()})
        # Start with the production NVIDIA selector; try alternatives only when
        # a route fails to execute. A parity mismatch is retained, never hidden.
        selected = None
        selectors = snapshot.get('device_selectors') or GPU_DEVICES
        for index, device in enumerate(selectors):
            prefix = 'device_{}'.format(index)
            checkpoint(prefix + '_selector', device)
            identity = GPUPlan(CreativeHDRSettings(), white, device)
            try:
                value = attempt(prefix + '_identity', lambda: command(identity.raw_command(
                                ffmpeg, width, height), payload))
            finally:
                identity.close()
            if value is None:
                continue
            checkpoint(prefix + '_identity', compare_bytes(payload, value, shape, True))
            selected = device
            plan = GPUPlan(grade, white, device)
            try:
                actual = attempt('opencl_grade_float', lambda: command(plan.raw_command(
                                 ffmpeg, width, height), payload))
                if actual is not None and expected is not None:
                    checkpoint('opencl_grade_float', compare_bytes(expected, actual, shape, True))
                    view = LINEAR_BT2020_TAGS + ',' + viewing_filter(white)
                    viewed_cpu = attempt('isolated_cpu_view', lambda: command(
                        cpu_command(ffmpeg, width, height, view, 'rgb24'), expected))
                    viewed_gpu = attempt('isolated_gpu_view', lambda: command(
                        cpu_command(ffmpeg, width, height, view, 'rgb24'), actual))
                    if viewed_cpu is not None and viewed_gpu is not None:
                        checkpoint('isolated_view_transform', compare_bytes(viewed_cpu, viewed_gpu,
                                                                          (height, width, 3)))
                gpu_pair = attempt('opencl_preview', lambda: command(render_command(
                                   ffmpeg, width, height, grade, white, gpu_plan=plan), payload))
                if gpu_pair is not None and cpu_pair is not None:
                    checkpoint('captured_preview', preview_comparison(cpu_pair, gpu_pair, width, height))
            finally:
                plan.close()
            break
        report['selected_device'] = selected
        report['complete'] = True
        report['status'] = 'checks_finished'  # Not a claim that the checks passed.
        if selected is None:
            report['summary'] = 'No tested GPU route executed successfully. Device errors are recorded.'
        else:
            pair = stages.get('captured_preview', {}).get('whole_pair', {})
            report['summary'] = ('Preview comparison passed for this diagnostic snapshot.'
                                 if pair.get('matches_existing_preview_tolerance') is True else
                                 'GPU executed, but the preview comparison failed or could not complete. Inspect recorded stages.')
    except DiagnosticCancelled:
        report['status'] = 'cancelled'
        report['summary'] = 'Diagnostic cancelled; any completed stages are retained.'
    except Exception as exc:
        report['complete'] = True
        report['status'] = 'diagnostic_error'
        report['error'] = '{}: {}'.format(type(exc).__name__, exc)
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        try:
            if report['status'] == 'cancelled':
                destination = destination.with_name(destination.name.replace('_READY.json', '_INTERRUPTED.json'))
            save_report(destination, report)
            if partial.exists():
                partial.unlink()
            events.put(('finished', str(destination), report))
        except Exception as exc:
            events.put(('save_error', str(exc)))
    return report


def build_session(window, repo):
    """Add visible controls; no interception of internal FFmpeg callbacks needed."""
    from PyQt5 import QtCore, QtGui, QtWidgets
    engine = window.preview.engine
    toolbar = QtWidgets.QGroupBox('GPU diagnostics v2', window)
    layout = QtWidgets.QVBoxLayout(toolbar)
    buttons = QtWidgets.QHBoxLayout()
    run_frame = QtWidgets.QPushButton('Run diagnostics on this frame')
    selftest = QtWidgets.QPushButton('Hardware self-test')
    open_report = QtWidgets.QPushButton('Open report folder')
    run_frame.setEnabled(False)
    open_report.setEnabled(False)
    for button in (run_frame, selftest, open_report):
        buttons.addWidget(button)
    layout.addLayout(buttons)
    label = QtWidgets.QLabel('Hardware self-test starts automatically. No video export.')
    label.setWordWrap(True)
    layout.addWidget(label)
    window.layout().insertWidget(0, toolbar)
    events, cancelled = queue.Queue(), threading.Event()
    state = {'busy': False, 'thread': None, 'report_path': None}

    def log(text):
        window.log_output.appendPlainText('[GPU diagnostics v2] ' + text)

    def start(snapshot):
        if state['busy']:
            return
        report = new_report(repo, snapshot['scope'])
        path = repo / ('hdr_gpu_' + snapshot['scope'] + '_' + report['utc'] + '_READY.json')
        state['busy'] = True
        cancelled.clear()
        label.setText('  RUNNING: ' + snapshot['scope'] + ' — no completed JSON yet.')
        log('Started {}. YouTube/export settings are unchanged.'.format(snapshot['scope']))
        thread = threading.Thread(target=investigate,
                  args=(snapshot, report, path, events, cancelled), daemon=True)
        state['thread'] = thread
        thread.start()

    def run_current():
        try:
            window.preview.stop_playback()
            start(snapshot_from_engine(engine))
        except Exception as exc:
            label.setText('  ' + str(exc))
            log(str(exc))

    def run_selftest():
        from conversion_paths import find_ffmpeg
        try:
            start(synthetic_snapshot(find_ffmpeg()))
        except Exception as exc:
            label.setText('  Could not start self-test: ' + str(exc))
            log(str(exc))

    def poll():
        try:
            while True:
                item = events.get_nowait()
                if item[0] == 'progress':
                    label.setText('  RUNNING: ' + item[1])
                    log('Recorded: ' + item[1])
                elif item[0] == 'finished':
                    state['busy'] = False
                    state['report_path'] = Path(item[1])
                    report = item[2]
                    open_report.setEnabled(True)
                    heading = 'INTERRUPTED — ' if report['status'] == 'cancelled' else 'REPORT READY — '
                    label.setText(heading + state['report_path'].name)
                    label.setToolTip(str(state['report_path']))
                    log('Report ready: ' + item[1])
                    log(report.get('summary', report.get('error', report['status'])))
                    log('Attach the report named above. Current-frame reports are most useful for a scene-specific problem.')
                elif item[0] == 'save_error':
                    state['busy'] = False
                    label.setText('  Report save failed: ' + item[1])
                    log(label.text())
        except queue.Empty:
            pass
        can_capture = (engine.source is not None and not engine.paused
                       and engine._key() in engine.cache)
        run_frame.setEnabled(not state['busy'] and can_capture)
        selftest.setEnabled(not state['busy'] and not engine.paused)

    run_frame.clicked.connect(run_current)
    selftest.clicked.connect(run_selftest)
    open_report.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(
        QtCore.QUrl.fromLocalFile(str(state['report_path'].parent))))
    engine.message.connect(lambda text: log(text) if text.startswith('GPU preview') else None)
    engine.failed.connect(lambda text: log('Preview error: ' + text))
    timer = QtCore.QTimer(window)
    timer.timeout.connect(poll)
    timer.start(150)
    QtCore.QTimer.singleShot(200, lambda: run_selftest() if not cancelled.is_set() else None)
    log('Open your SDR video, choose the frame/grade, then click Run diagnostics on this frame.')
    log('This explicitly tests the GPU even if the normal preview is using CPU fallback.')
    # Return strong references and explicit hooks for cleanup/tests.
    return dict(state=state, cancel=cancelled, timer=timer, toolbar=toolbar,
                start=start, run_current=run_current, run_selftest=run_selftest,
                poll=poll, label=label, frame_action=run_frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path)
    parser.add_argument('--headless', action='store_true', help='Run only the synthetic self-test without Qt')
    args = parser.parse_args()
    adjacent = Path(__file__).resolve().parent
    repo = (args.repo or (adjacent if (adjacent / 'SDR_to_HDR_cross_platform.py').is_file()
                         else Path.cwd())).resolve()
    for name in ('SDR_to_HDR_cross_platform.py', 'gpu_float_transport.py'):
        if not (repo / name).is_file():
            parser.error('Save this script beside the updated application source, or pass --repo PATH. Missing: ' + name)
    sys.path.insert(0, str(repo))
    if args.headless:
        from conversion_paths import find_ffmpeg
        snapshot = synthetic_snapshot(find_ffmpeg())
        report = new_report(repo, 'hardware_selftest')
        path = repo / ('hdr_gpu_hardware_selftest_' + report['utc'] + '_READY.json')
        class ConsoleEvents:
            def put(self, item):
                print(item[0] + ': ' + str(item[1]), flush=True)
        investigate(snapshot, report, path, ConsoleEvents(), threading.Event())
        print('Attach this report:', path, flush=True)
        return 0 if path.exists() else 1
    try:
        from PyQt5 import QtWidgets
        from SDR_to_HDR_cross_platform import SDRtoHDRConverter
    except ImportError as exc:
        parser.error('Use your application Python environment, or run --headless for a hardware report. ' + str(exc))
    app = QtWidgets.QApplication([sys.argv[0]])
    window = SDRtoHDRConverter()
    window.setWindowTitle('HDR Video Studio — GPU DIAGNOSTICS v2')
    session = build_session(window, repo)
    app.aboutToQuit.connect(session['cancel'].set)
    window.show()
    result = app.exec_()
    session['cancel'].set()
    thread = session['state']['thread']
    if thread is not None:
        thread.join(3)
    return result


if __name__ == '__main__':
    sys.exit(main())
