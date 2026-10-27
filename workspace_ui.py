"""Preview-first workspace; the existing export engine and widget contracts remain."""
from pathlib import Path
import re
from PyQt5 import QtCore, QtGui, QtWidgets
from creative_hdr_ui import CreativeHDRPanel
from preview_widget import GradingPreview
from youtube_hdr_export import YOUTUBE_QUALITY_PROFILES

STYLE = '''
QWidget#HDRWorkspace { background:#1a1e25; color:#e6eaf0; }
QWidget#HDRWorkspace QWidget { color:#e6eaf0; font-size:13px; }
QLabel { background:transparent; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {
 background:#242a34; color:#e6eaf0; border:1px solid #465161; border-radius:4px; padding:5px;
 selection-background-color:#37675c; }
QPushButton, QToolButton { background:#303946; border:1px solid #516071; border-radius:4px; padding:6px 10px; }
QPushButton:hover, QToolButton:hover { background:#405064; }
QPushButton#PrimaryExport { background:#226550; border:1px solid #48ac88; font-weight:600; }
QPushButton#PrimaryExport:hover { background:#2d8066; }
QWidget:disabled { color:#85909f; }
QGroupBox { border:1px solid #414b5a; border-radius:6px; margin-top:12px; padding-top:12px; }
QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 4px; }
QTabWidget::pane { border:1px solid #414b5a; background:#1e242d; }
QTabBar::tab { background:#252c36; padding:8px 12px; border-bottom:2px solid #303946; }
QTabBar::tab:selected { background:#303b49; border-bottom:2px solid #58c69d; }
QScrollArea, QScrollArea > QWidget > QWidget { background:#1e242d; border:none; }
QSlider::groove:horizontal { height:5px; background:#465161; border-radius:2px; }
QSlider::handle:horizontal { background:#77cdb0; width:12px; margin:-5px 0; border-radius:5px; }
QSlider::sub-page:horizontal { background:#4d9c80; }
QSplitter::handle { background:#3b4552; }
QProgressBar { border:1px solid #465161; border-radius:3px; max-height:8px; background:#242a34; }
QProgressBar::chunk { background:#4eac8a; }
'''


def _label(text):
    label = QtWidgets.QLabel(text)
    label.setWordWrap(True)
    return label


def _scroll(page):
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
    scroll.setWidget(page)
    return scroll


def _check(text, checked=False):
    box = QtWidgets.QCheckBox(text)
    box.setChecked(checked)
    return box


def build_workspace(w):
    w.setObjectName('HDRWorkspace')
    w.setWindowTitle('HDR Video Studio — Preview, Grade and Export')
    w.setStyleSheet(STYLE)
    screen = QtWidgets.QApplication.primaryScreen()
    area = screen.availableGeometry() if screen else QtCore.QRect(0, 0, 1440, 1000)
    w.resize(min(1320, int(area.width() * .94)), min(880, int(area.height() * .90)))
    w.setMinimumSize(880, 590)
    root = QtWidgets.QVBoxLayout(w)
    root.setContentsMargins(16, 12, 16, 12)
    heading = QtWidgets.QHBoxLayout()
    title = QtWidgets.QLabel('HDR Video Studio')
    title.setStyleSheet('font-size:22px; font-weight:600;')
    heading.addWidget(title)
    heading.addStretch()
    heading.addWidget(_label('01  Source     02  Preview & grade     03  Export'))
    root.addLayout(heading)
    w.source_group = QtWidgets.QWidget()
    inputs = QtWidgets.QGridLayout(w.source_group)
    inputs.setContentsMargins(0, 0, 0, 4)
    w.input_path, w.output_path = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
    w.input_path.setPlaceholderText('Choose an SDR video, or a folder for batch conversion')
    w.output_path.setPlaceholderText('Choose the destination folder')
    w.input_btn = QtWidgets.QPushButton('Open video…')
    w.output_btn = QtWidgets.QPushButton('Output folder…')
    w.input_btn.clicked.connect(w.select_input_file)
    w.output_btn.clicked.connect(w.select_output_file)
    w.batch_mode = _check('Batch folder')
    inputs.addWidget(QtWidgets.QLabel('Source'), 0, 0)
    inputs.addWidget(w.input_path, 0, 1)
    inputs.addWidget(w.input_btn, 0, 2)
    inputs.addWidget(w.batch_mode, 0, 3)
    inputs.addWidget(QtWidgets.QLabel('Save to'), 1, 0)
    inputs.addWidget(w.output_path, 1, 1)
    inputs.addWidget(w.output_btn, 1, 2, 1, 2)
    inputs.setColumnStretch(1, 1)
    root.addWidget(w.source_group)
    w.hdr_reference_label = QtWidgets.QLabel('Reference white (nits)')
    w.hdr_reference_nits = QtWidgets.QSpinBox()
    w.hdr_reference_nits.setRange(80, 400)
    w.hdr_reference_nits.setValue(200)
    w.hdr_reference_nits.setKeyboardTracking(False)
    w.hdr_reference_nits.setToolTip('200 nits is the standard mapping used by this app; independent of the creative peak ceiling.')
    w.creative_panel = CreativeHDRPanel(w)
    w.preview = GradingPreview(w.creative_panel, w.hdr_reference_nits, w)
    w.controls_tabs = QtWidgets.QTabWidget()
    w.controls_tabs.setMinimumWidth(350)
    w.grade_page = QtWidgets.QWidget()
    grade_layout = QtWidgets.QVBoxLayout(w.grade_page)
    grade_layout.addWidget(_label('Select a frame and adjust the controls. The preview updates automatically; no test export is required.'))
    reference = QtWidgets.QHBoxLayout()
    reference.addWidget(w.hdr_reference_label)
    reference.addWidget(w.hdr_reference_nits)
    grade_layout.addLayout(reference)
    grade_layout.addWidget(w.creative_panel)
    grade_layout.addStretch()
    w.controls_tabs.addTab(_scroll(w.grade_page), 'Grade')
    w.export_page = QtWidgets.QWidget()
    exports = QtWidgets.QVBoxLayout(w.export_page)
    youtube = QtWidgets.QGroupBox('YouTube HDR export')
    yl = QtWidgets.QVBoxLayout(youtube)
    yl.addWidget(_label('10-bit HEVC · BT.2020/PQ · CPU x265\nSettings here apply to Export YouTube HDR.'))
    w.youtube_quality = QtWidgets.QComboBox()
    w.youtube_quality.addItems(list(YOUTUBE_QUALITY_PROFILES))
    w.youtube_quality.setCurrentText('High')
    w.youtube_quality.setToolTip('Balanced: CRF 20/medium; High: CRF 16/medium; Maximum: CRF 12/slow.')
    yl.addWidget(QtWidgets.QLabel('Quality'))
    yl.addWidget(w.youtube_quality)
    w.youtube_analyse_light = _check('Measure content-light metadata', True)
    w.youtube_analyse_light.setToolTip('Creative mode always measures MaxCLL/MaxFALL after the grade.')
    w.youtube_normalize_audio = _check('Normalize audio (−16 LUFS)')
    w.youtube_normalize_audio.setToolTip('Optional two-pass loudness normalization, -1.5 dBTP target; may affect dynamics.')
    yl.addWidget(w.youtube_analyse_light)
    yl.addWidget(w.youtube_normalize_audio)
    exports.addWidget(youtube)
    custom = QtWidgets.QGroupBox('Custom export')
    cl = QtWidgets.QVBoxLayout(custom)
    cl.addWidget(_label('Settings here apply only to Export custom.\nCreative grading requires 10-bit output.'))
    w.bit_depth_label = QtWidgets.QLabel('Output bit depth')
    w.bit_depth = QtWidgets.QComboBox()
    w.bit_depth.addItems(['10', '8'])
    line = QtWidgets.QHBoxLayout()
    line.addWidget(w.bit_depth_label)
    line.addWidget(w.bit_depth)
    cl.addLayout(line)
    w.gpu_checkbox = _check('GPU encoding (NVENC)')
    w.audio_passthrough = _check('Copy original audio')
    w.gpu_checkbox.setToolTip('Custom export only. Grading remains CPU-based; this NVENC path does not embed x265 static mastering SEI.')
    w.audio_passthrough.setToolTip('Only when the source audio codec is supported by the destination container.')
    cl.addWidget(w.gpu_checkbox)
    cl.addWidget(w.audio_passthrough)
    w.cq_label = QtWidgets.QLabel('Constant quality (lower = higher quality)')
    cl.addWidget(w.cq_label)
    w.cq_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    w.cq_slider.setRange(0, 51)
    w.cq_slider.setValue(23)
    w.cq_value_label = QtWidgets.QLabel('23')
    w.cq_slider.valueChanged.connect(lambda value: w.cq_value_label.setText(str(value)))
    crf_row = QtWidgets.QHBoxLayout()
    crf_row.addWidget(w.cq_slider)
    crf_row.addWidget(w.cq_value_label)
    cl.addLayout(crf_row)
    w.color_processing_label = _label('')
    cl.addWidget(w.color_processing_label)
    exports.addWidget(custom)
    metadata_toggle = QtWidgets.QToolButton()
    metadata_toggle.setText('Advanced metadata')
    metadata_toggle.setCheckable(True)
    metadata_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
    metadata_toggle.setArrowType(QtCore.Qt.RightArrow)
    metadata = QtWidgets.QWidget()
    ml = QtWidgets.QVBoxLayout(metadata)
    w.embed_metadata = _check('Embed static HDR10 metadata (x265)', True)
    w.generate_metadata = _check('Generate static metadata')
    w.generate_metadata.hide()
    w.override_metadata = _check('Override MaxCLL / MaxFALL')
    w.max_cll_input, w.max_fall_input = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
    w.max_cll_input.setPlaceholderText('MaxCLL (nits)')
    w.max_fall_input.setPlaceholderText('MaxFALL (nits)')
    ml.addWidget(w.embed_metadata)
    ml.addWidget(w.override_metadata)
    ml.addWidget(w.max_cll_input)
    ml.addWidget(w.max_fall_input)
    ml.addWidget(_label('Creative grading uses freshly measured content-light metadata, not manual values.'))
    exports.addWidget(metadata_toggle)
    exports.addWidget(metadata)
    metadata.hide()
    def show_metadata(show):
        metadata.setVisible(show)
        metadata_toggle.setArrowType(QtCore.Qt.DownArrow if show else QtCore.Qt.RightArrow)
    metadata_toggle.toggled.connect(show_metadata)
    exports.addStretch()
    w.controls_tabs.addTab(_scroll(w.export_page), 'Export settings')
    w.log_output = QtWidgets.QPlainTextEdit()
    w.log_output.setReadOnly(True)
    w.log_output.setMaximumBlockCount(3000)
    w.controls_tabs.addTab(w.log_output, 'Log')
    w.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
    w.splitter.addWidget(w.preview)
    w.splitter.addWidget(w.controls_tabs)
    w.splitter.setStretchFactor(0, 3)
    w.splitter.setStretchFactor(1, 2)
    w.splitter.setSizes([760, 400])
    root.addWidget(w.splitter, 1)
    w.progress = QtWidgets.QProgressBar()
    w.progress.setRange(0, 1)
    w.progress.setValue(0)
    w.progress.setTextVisible(False)
    root.addWidget(w.progress)
    footer = QtWidgets.QHBoxLayout()
    w.status_label = _label('Ready — open a source video to start.')
    footer.addWidget(w.status_label, 1)
    w.cancel_btn = QtWidgets.QPushButton('Cancel')
    w.cancel_btn.clicked.connect(w.cancel_conversion)
    w.cancel_btn.setEnabled(False)
    w.start_btn = QtWidgets.QPushButton('Export custom')
    w.start_btn.clicked.connect(w.start_conversion)
    w.youtube_btn = QtWidgets.QPushButton('Export YouTube HDR')
    w.youtube_btn.setObjectName('PrimaryExport')
    w.youtube_btn.clicked.connect(w.start_youtube_export)
    footer.addWidget(w.cancel_btn)
    footer.addWidget(w.start_btn)
    footer.addWidget(w.youtube_btn)
    root.addLayout(footer)
    w.progress_detail = _label('Original source is never modified. Preview is a display approximation; exports retain 10-bit PQ.')
    root.addWidget(w.progress_detail)
    w.preview_mode = _check('Live preview', True)
    w.preview_mode.hide()
    w.bit_depth.currentTextChanged.connect(w.update_hdr_controls)
    w.override_metadata.stateChanged.connect(w.toggle_override_fields)
    w.creative_panel.toggled.connect(w.toggle_override_fields)
    w.update_hdr_controls(w.bit_depth.currentText())
    w.source_timer = QtCore.QTimer(w)
    w.source_timer.setSingleShot(True)
    w.source_timer.setInterval(400)
    w.source_timer.timeout.connect(lambda: _load_source(w))
    w.input_path.textChanged.connect(lambda _: _source_changed(w))
    w.batch_mode.toggled.connect(lambda _: _batch_changed(w))
    w.log_output.textChanged.connect(lambda: _progress_log(w))


def _source_changed(w):
    if getattr(w, 'export_active', False):
        return
    w.preview.invalidate('Loading source preview...' if not w.batch_mode.isChecked() else 'Batch mode: choose a preview clip.')
    w.source_timer.start()


def _batch_changed(w):
    w.input_btn.setText('Open folder…' if w.batch_mode.isChecked() else 'Open video…')
    _source_changed(w)


def _load_source(w):
    if w.batch_mode.isChecked() or getattr(w, 'export_active', False):
        return
    path = w.input_path.text().strip()
    if path:
        w.preview.load_source(path)


def _progress_log(w):
    last = w.log_output.document().lastBlock().text()
    if last.startswith('frame='):
        frame = re.search(r'frame=\s*(\d+)', last)
        position = re.search(r'time=([^ ]+)', last)
        if frame:
            w.progress_detail.setText('Encoding frame {}{}'.format(
                frame.group(1), ' • output time ' + position.group(1) if position else ''))
    elif last.startswith(('Analysed ', 'Measured HDR', 'Measured ', 'Verified ', 'Audio input')):
        w.progress_detail.setText(last)


def set_exporting(w, active):
    w.export_active = active
    w.source_timer.stop()
    w.source_group.setEnabled(not active)
    w.grade_page.setEnabled(not active)
    w.export_page.setEnabled(not active)
    w.preview.set_exporting(active)
    w.cancel_btn.setEnabled(active)
    w.progress.setRange(0, 0 if active else 1)
    if not active:
        w.progress.setValue(0)
