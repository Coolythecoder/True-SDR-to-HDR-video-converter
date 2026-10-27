"""Asynchronous, latest-request-wins live preview. No GUI-thread FFmpeg waits."""
from collections import OrderedDict
import json
import time
from pathlib import Path
from PyQt5 import QtCore, QtGui, QtWidgets
from conversion_paths import find_ffmpeg
from hdr_processing import find_ffprobe
from creative_hdr import CreativeHDRSettings
from grading_preview import source_from_probe, probe_command, decode_command, render_command


class PreviewEngine(QtCore.QObject):
    sourceReady = QtCore.pyqtSignal(object)
    frameReady = QtCore.pyqtSignal(object, object, float, float)
    message = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.source = None
        self.ffmpeg = None
        self.process = None
        self.phase = ''
        self.cache = OrderedDict()
        self.seconds = 0.0
        self.bound = 640
        self.grade = CreativeHDRSettings()
        self.white_nits = 200
        self.revision = 0
        self.closed = False
        self.paused = False
        self._processes = set()
        self._decode_key = None
        self.debounce = QtCore.QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(160)
        self.debounce.timeout.connect(self._schedule)

    def cancel(self):
        self.debounce.stop()
        process, self.process = self.process, None
        self.phase = ''
        if process is not None:
            process.kill()

    def invalidate(self, reason='Select an SDR video to preview.'):
        self.revision += 1
        self.source = None
        self.cache.clear()
        self.cancel()
        self.message.emit(reason)

    def load(self, path):
        self.invalidate('Reading source information...')
        if self.closed or self.paused:
            return
        candidate = Path(path).expanduser()
        if not candidate.is_file():
            self.failed.emit('Select an existing video file, not a folder.')
            return
        self.ffmpeg = find_ffmpeg()
        ffprobe = find_ffprobe(self.ffmpeg) if self.ffmpeg else None
        if not self.ffmpeg or not ffprobe:
            self.failed.emit('Preview needs FFmpeg and FFprobe on PATH.')
            return
        resolved = str(candidate.resolve())

        def ready(data):
            self.source = source_from_probe(resolved, json.loads(data.decode('utf-8')))
            self.seconds = 0
            self.sourceReady.emit(self.source)
            self.debounce.start()
        self._launch(probe_command(ffprobe, resolved), 'probe', ready, limit=2 * 1024 * 1024)

    def request(self, grade=None, white_nits=None, seconds=None, bound=None):
        if grade is not None:
            self.grade = grade
        if white_nits is not None:
            self.white_nits = white_nits
        if seconds is not None:
            self.seconds = max(0, float(seconds))
        if bound is not None:
            self.bound = bound
        self.revision += 1
        if not self.closed and not self.paused:
            self.debounce.start()

    def _key(self):
        fps = self.source.fps
        seconds = min(self.seconds, max(0, self.source.duration - 1 / fps))
        frame = max(0, int(round(seconds * fps)))
        return self.source.path, frame, self.bound

    def _schedule(self):
        if self.closed or self.paused or self.source is None:
            return
        try:
            self.grade.validate(self.white_nits)
            key = self._key()
            width, height = self.source.proxy_size(self.bound)
            if self.phase == 'probe':
                return
            if self.phase == 'decode' and self._decode_key == key:
                return
            self.cancel()
            if key not in self.cache:
                self._decode_key = key
                self.message.emit('Decoding preview frame...')
                cmd = decode_command(self.ffmpeg, self.source, key[1] / self.source.fps, self.bound)

                def decoded(data):
                    if len(data) != width * height * 12:
                        raise ValueError('No complete preview frame at this time. Move back one frame.')
                    self.cache[key] = data
                    self.cache.move_to_end(key)
                    while len(self.cache) > 3:
                        self.cache.popitem(last=False)
                    self._schedule()
                self._launch(cmd, 'decode', decoded, limit=width * height * 12 + 1024)
                return
            self.cache.move_to_end(key)
            revision = self.revision
            started = time.monotonic()
            self.message.emit('Updating grade...')

            def rendered(data):
                if revision != self.revision or key != self._key():
                    self.debounce.start()
                    return
                if len(data) != width * height * 6:
                    raise ValueError('Incomplete preview output from FFmpeg.')
                pair = QtGui.QImage(data, width * 2, height, width * 6,
                                    QtGui.QImage.Format_RGB888).copy()
                if pair.isNull():
                    raise ValueError('Could not display the preview frame.')
                if hasattr(pair, 'setColorSpace') and hasattr(QtGui, 'QColorSpace'):
                    pair.setColorSpace(QtGui.QColorSpace(QtGui.QColorSpace.SRgb))
                self.frameReady.emit(pair.copy(0, 0, width, height),
                                     pair.copy(width, 0, width, height),
                                     key[1] / self.source.fps,
                                     time.monotonic() - started)
            self._launch(render_command(self.ffmpeg, width, height, self.grade, self.white_nits),
                         'render', rendered, payload=self.cache[key], limit=width * height * 6 + 1024)
        except (ValueError, OSError, TypeError, KeyError) as exc:
            self.cancel()
            self.failed.emit(str(exc))

    def _launch(self, command, phase, callback, payload=None, limit=8 * 1024 * 1024):
        process = QtCore.QProcess(self)
        self.process = process
        self._processes.add(process)
        self.phase = phase
        state = {'stdout': bytearray(), 'stderr': bytearray(), 'error': '', 'done': False}
        timer = QtCore.QTimer(process)
        timer.setSingleShot(True)

        def read_out():
            data = bytes(process.readAllStandardOutput())
            if len(state['stdout']) + len(data) > limit:
                state['error'] = 'Preview output exceeded its safety limit.'
                process.kill()
            elif not state['error']:
                state['stdout'].extend(data)

        def read_error():
            state['stderr'].extend(bytes(process.readAllStandardError()))
            del state['stderr'][:-16384]

        def complete(code=0, status=QtCore.QProcess.NormalExit):
            if state['done']:
                return
            state['done'] = True
            timer.stop()
            read_out()
            read_error()
            current = process is self.process
            if current:
                self.process = None
                self.phase = ''
            self._processes.discard(process)
            process.deleteLater()
            if not current or self.closed or self.paused:
                return
            error = state['error']
            if code or status != QtCore.QProcess.NormalExit:
                error = error or state['stderr'].decode('utf-8', errors='replace')[-1200:]
                error = error or 'FFmpeg could not render this preview.'
            if error:
                self.failed.emit(error)
                return
            try:
                callback(bytes(state['stdout']))
            except (ValueError, TypeError, KeyError, OSError, IndexError) as exc:
                self.failed.emit(str(exc))

        def failed(error):
            if error == QtCore.QProcess.FailedToStart:
                state['error'] = 'Cannot start preview tool: ' + process.errorString()
                complete(-1)

        def timeout():
            state['error'] = 'Preview timed out. Try a lower preview resolution or another frame.'
            process.kill()

        def started():
            if payload is not None:
                process.write(payload)
            process.closeWriteChannel()
        process.readyReadStandardOutput.connect(read_out)
        process.readyReadStandardError.connect(read_error)
        process.errorOccurred.connect(failed)
        process.finished.connect(complete)
        process.started.connect(started)
        timer.timeout.connect(timeout)
        process.start(command[0], command[1:])
        timer.start(30000)

    def set_paused(self, paused):
        self.paused = paused
        if paused:
            self.cancel()
        elif self.source:
            self.request()

    def shutdown(self):
        self.closed = True
        self.cancel()
        for process in list(self._processes):
            process.kill()
            process.waitForFinished(1500)


class CompareCanvas(QtWidgets.QWidget):
    """Before, after and draggable split views of synchronized frames."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.before = self.after = None
        self.mode = 'Split comparison'
        self.split = .5
        self.setMinimumSize(260, 160)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setToolTip('Drag the divider: before grade on the left, after grade on the right.')
        self.setAccessibleName('Before and after colour grading preview')

    def clear(self):
        self.before = self.after = None
        self.update()

    def set_frames(self, before, after):
        self.before, self.after = before, after
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor('#14171b'))
        if self.before is None:
            painter.setPen(QtGui.QColor('#c0c6cf'))
            painter.drawText(self.rect().adjusted(20, 20, -20, -20),
                             QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap,
                             'Open an SDR video to begin.\nChoose a frame, then adjust the grade.')
            return
        size = self.before.size().scaled(self.size(), QtCore.Qt.KeepAspectRatio)
        target = QtCore.QRect((self.width() - size.width()) // 2,
                              (self.height() - size.height()) // 2,
                              size.width(), size.height())
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        painter.drawImage(target, self.before if self.mode == 'Before grade' else self.after)
        if self.mode == 'Split comparison':
            x = target.left() + int(target.width() * self.split)
            painter.save()
            painter.setClipRect(QtCore.QRect(target.left(), target.top(), x-target.left(), target.height()))
            painter.drawImage(target, self.before)
            painter.restore()
            painter.setPen(QtGui.QPen(QtGui.QColor('#ffffff'), 2))
            painter.drawLine(x, target.top(), x, target.bottom())
        painter.fillRect(10, 10, 205, 24, QtGui.QColor(0, 0, 0, 180))
        painter.setPen(QtCore.Qt.white)
        label = 'BEFORE   |   AFTER' if self.mode == 'Split comparison' else self.mode.upper()
        painter.drawText(18, 27, label)

    def mousePressEvent(self, event):
        self.mouseMoveEvent(event)

    def mouseMoveEvent(self, event):
        if self.before is not None and event.buttons() & QtCore.Qt.LeftButton:
            size = self.before.size().scaled(self.size(), QtCore.Qt.KeepAspectRatio)
            left = (self.width() - size.width()) / 2
            self.split = min(1, max(0, (event.x() - left) / max(size.width(), 1)))
            self.update()


class GradingPreview(QtWidgets.QWidget):
    def __init__(self, panel, reference_white, parent=None):
        super().__init__(parent)
        self.panel = panel
        self.reference_white = reference_white
        self.engine = PreviewEngine(self)
        self.source = None
        self.playing = False
        self._suspended = False
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel('LIVE GRADE PREVIEW')
        title.setStyleSheet('font-weight:600; letter-spacing:1px;')
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(['Split comparison', 'After grade', 'Before grade'])
        self.quality = QtWidgets.QComboBox()
        for width, name in ((480, 'Fast (480)'), (640, 'Balanced (640)'), (960, 'Detail (960)')):
            self.quality.addItem(name, width)
        self.quality.setCurrentIndex(1)
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.mode)
        header.addWidget(self.quality)
        layout.addLayout(header)
        self.canvas = CompareCanvas(self)
        layout.addWidget(self.canvas, 1)
        self.timeline = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.timeline.setRange(0, 10000)
        self.timeline.setEnabled(False)
        self.timeline.setAccessibleName('Preview timeline')
        layout.addWidget(self.timeline)
        transport = QtWidgets.QHBoxLayout()
        self.previous = QtWidgets.QPushButton('−1 frame')
        self.play = QtWidgets.QPushButton('Play proxy')
        self.next = QtWidgets.QPushButton('+1 frame')
        self.refresh = QtWidgets.QPushButton('Refresh')
        self.position = QtWidgets.QDoubleSpinBox()
        self.position.setDecimals(3)
        self.position.setSuffix(' s')
        self.position.setRange(0, 0)
        self.time_label = QtWidgets.QLabel('/ 0.00 s')
        for widget in (self.previous, self.play, self.next, self.position, self.time_label):
            transport.addWidget(widget)
        transport.addStretch()
        transport.addWidget(self.refresh)
        layout.addLayout(transport)
        self.info = QtWidgets.QLabel('No preview source selected.')
        self.info.setWordWrap(True)
        self.status = QtWidgets.QLabel('Open a source video. Batch mode uses a separately chosen preview clip.')
        self.status.setWordWrap(True)
        self.pick = QtWidgets.QPushButton('Choose preview clip…')
        layout.addWidget(self.info)
        layout.addWidget(self.status)
        layout.addWidget(self.pick)
        note = QtWidgets.QLabel(
            'SDR tone-mapped approximation, not native HDR. Before = accurate conversion without a creative grade. Both views use the same '
            'fixed 1,000-nit viewing scale. Preview resolution and playback do not affect export. '
            'Proxy playback is silent and performance-limited; judge final HDR brightness on an HDR display.')
        note.setWordWrap(True)
        note.setStyleSheet('color:#9ca8b7; font-size:11px;')
        layout.addWidget(note)
        self.tick = QtCore.QTimer(self)
        self.tick.setSingleShot(True)
        self.tick.timeout.connect(self._advance)
        self.engine.sourceReady.connect(self._source_ready)
        self.engine.frameReady.connect(self._frame_ready)
        self.engine.message.connect(self.status.setText)
        self.engine.failed.connect(self._error)
        self.mode.currentTextChanged.connect(self._mode)
        self.quality.currentIndexChanged.connect(self._request)
        self.timeline.valueChanged.connect(self._seek_slider)
        self.position.valueChanged.connect(self._seek_position)
        self.previous.clicked.connect(lambda: self.step(-1))
        self.next.clicked.connect(lambda: self.step(1))
        self.play.clicked.connect(self._play)
        self.refresh.clicked.connect(self._refresh)
        self.pick.clicked.connect(self._pick)
        panel.settingsChanged.connect(self._request)
        reference_white.valueChanged.connect(self._request)
        self._transport(False)

    def _transport(self, enabled):
        for widget in (self.timeline, self.position, self.previous, self.next, self.play, self.refresh):
            widget.setEnabled(enabled)

    def load_source(self, path):
        self.stop_playback()
        self.canvas.clear()
        self.source = None
        self._transport(False)
        self.info.setText('Loading ' + Path(path).name)
        self.engine.request(self.panel.settings(), self.reference_white.value())
        self.engine.load(path)

    def invalidate(self, message='Choose a single clip for the preview.'):
        self.stop_playback()
        self.source = None
        self.canvas.clear()
        self._transport(False)
        self.info.setText(message)
        self.engine.invalidate(message)

    def _source_ready(self, source):
        self.source = source
        self.position.blockSignals(True)
        self.position.setRange(0, max(0, source.duration - 1 / source.fps))
        self.position.setValue(0)
        self.position.setSingleStep(1 / source.fps)
        self.position.blockSignals(False)
        self.timeline.blockSignals(True)
        self.timeline.setValue(0)
        self.timeline.blockSignals(False)
        self.time_label.setText('/ {:.2f} s'.format(source.duration))
        assumption = ' • Missing tags assumed BT.709 / limited range' if source.assumptions else ''
        self.info.setText('{}  •  {} × {}  •  {:.3f} fps  •  {} range{}'.format(
            Path(source.path).name, source.width, source.height, source.fps,
            source.profile['range'], assumption))
        self._transport(True)
        self._request()

    def _request(self, *_):
        if self._suspended:
            return
        self.engine.request(self.panel.settings(), self.reference_white.value(),
                            self.position.value(), self.quality.currentData())

    def _refresh(self):
        self.engine.cache.clear()
        self._request()

    def _mode(self, mode):
        self.canvas.mode = mode
        self.canvas.update()

    def _seek_slider(self, value):
        if self.source:
            self.position.setValue(self.position.maximum() * value / 10000)

    def _seek_position(self, value):
        if self.source:
            self.timeline.blockSignals(True)
            self.timeline.setValue(round(value / max(self.position.maximum(), .001) * 10000))
            self.timeline.blockSignals(False)
            self._request()

    def step(self, direction):
        if self.source:
            self.stop_playback()
            self.position.setValue(self.position.value() + direction / self.source.fps)

    def _frame_ready(self, before, after, seconds, elapsed):
        self.canvas.set_frames(before, after)
        self.status.setText('Preview at {:.3f} s • updated in {:.2f} s • {}'.format(
            seconds, elapsed, 'Creative grade' if self.panel.isChecked() else 'Accurate conversion'))
        if self.playing:
            self.tick.start(50)

    def _play(self):
        if self.playing:
            self.stop_playback()
        elif self.source:
            self.playing = True
            self.play.setText('Pause')
            self._request()

    def _advance(self):
        if not self.playing or not self.source:
            return
        if self.position.value() >= self.position.maximum() - .001:
            self.stop_playback()
            return
        self.position.setValue(min(self.position.maximum(), self.position.value() + .25))

    def stop_playback(self):
        self.playing = False
        self.tick.stop()
        self.play.setText('Play proxy')

    def _pick(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Choose a clip to preview (does not change batch input)', '',
            'Video files (*.mp4 *.mkv *.mov)')
        if path:
            self.load_source(path)

    def _error(self, message):
        self.stop_playback()
        self.status.setText('Preview unavailable: ' + message)

    def set_exporting(self, active):
        self._suspended = active
        self.stop_playback()
        self.engine.set_paused(active)
        self.setEnabled(not active)
        if active:
            self.status.setText('Preview paused during export; the export uses a settings snapshot.')

    def shutdown(self):
        self.stop_playback()
        self.engine.shutdown()
