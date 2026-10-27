"""Grouped grading controls with linked sliders, precise values and change signal."""
from dataclasses import asdict
from PyQt5 import QtCore, QtWidgets
from creative_hdr import CreativeHDRSettings


class CreativeHDRPanel(QtWidgets.QGroupBox):
    settingsChanged = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__('Enable creative HDR grade', parent)
        self.setCheckable(True)
        self.setChecked(False)
        self._loading = False
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel('Adjust a slider and watch the preview. Uncheck this group to return to accurate conversion.')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.presets = QtWidgets.QComboBox()
        self.presets.addItems(['Custom', 'Gentle 400 nits', 'Creative 600 nits', 'Bright 1000 nits'])
        layout.addWidget(self.presets)
        self.tabs = QtWidgets.QTabWidget()
        self.controls, self.sliders = {}, {}
        definitions = [
            ('Highlights', [
                ('peak_nits', 'Peak ceiling (nits)', 200, 4000, 50, 0, 'Brightness ceiling, not the brightness of every white pixel.'),
                ('strength', 'Highlight expansion', 0, 1, .05, 2, 'Raise bright areas without globally raising midtones.'),
                ('knee', 'Highlight threshold', .5, .95, .05, 2, 'Fraction of linear reference white where expansion starts.'),
                ('rolloff', 'Highlight softness', .05, 1, .05, 2, 'Smooth shoulder approaching the brightness ceiling.'),
            ]),
            ('Tone', [
                ('exposure', 'Exposure (stops)', -2, 2, .1, 2, 'Global brightness. +1 doubles linear light.'),
                ('contrast', 'Contrast', .7, 1.4, .05, 2, 'Contrast around an 18% linear grey pivot.'),
                ('midtones', 'Midtones (stops)', -1, 1, .1, 2, 'Weighted middle-grey adjustment.'),
                ('shadows', 'Shadows (stops)', -2, 2, .1, 2, 'Dark-tone adjustment; black remains black.'),
                ('black_floor', 'Black lift (nits)', 0, 2, .05, 2, 'Explicitly lift the darkest pixels. Zero preserves black.'),
            ]),
            ('Colour', [
                ('saturation', 'Saturation', 0, 1.5, .05, 2, 'Colour intensity. 1 is neutral; 0 is monochrome.'),
                ('vibrance', 'Vibrance', -.5, 1, .05, 2, 'Primarily adjusts less saturated colours; not skin detection.'),
                ('temperature', 'Cool / warm', -1, 1, .05, 2, 'Creative white balance, not calibrated Kelvin.'),
                ('tint', 'Green / magenta', -1, 1, .05, 2, 'Creative tint adjustment.'),
            ]),
        ]
        defaults = asdict(CreativeHDRSettings())
        for tab_name, rows in definitions:
            page = QtWidgets.QWidget()
            page_layout = QtWidgets.QVBoxLayout(page)
            for key, label, low, high, step, decimals, tip in rows:
                page_layout.addWidget(QtWidgets.QLabel(label))
                row = QtWidgets.QHBoxLayout()
                control = QtWidgets.QDoubleSpinBox()
                control.setRange(low, high)
                control.setDecimals(decimals)
                control.setSingleStep(step)
                control.setValue(defaults[key])
                control.setKeyboardTracking(False)
                control.setFixedWidth(85)
                slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
                scale = 10 ** decimals
                slider.setRange(round(low * scale), round(high * scale))
                slider.setSingleStep(max(1, round(step * scale)))
                slider.setPageStep(max(1, round(step * scale * 4)))
                slider.setValue(round(defaults[key] * scale))
                slider.setAccessibleName(label)
                control.setAccessibleName(label + ' precise value')
                control.setToolTip(tip)
                slider.setToolTip(tip)
                reset = QtWidgets.QToolButton()
                reset.setText('↺')
                reset.setToolTip('Reset ' + label.lower())
                reset.clicked.connect(lambda _, c=control, v=defaults[key]: c.setValue(v))
                slider.valueChanged.connect(lambda value, c=control, s=scale: c.setValue(value / s))
                control.valueChanged.connect(lambda value, k=key, s=scale: self._changed(k, value, s))
                self.controls[key] = control
                self.sliders[key] = slider
                row.addWidget(slider, 1)
                row.addWidget(control)
                row.addWidget(reset)
                page_layout.addLayout(row)
            page_layout.addStretch()
            self.tabs.addTab(page, tab_name)
        layout.addWidget(self.tabs)
        self.reset_button = QtWidgets.QPushButton('Reset to accurate conversion')
        self.reset_button.clicked.connect(self.reset_neutral)
        layout.addWidget(self.reset_button)
        self.presets.activated[str].connect(self.apply_preset)
        self.toggled.connect(self._toggled)

    def _toggled(self, _value):
        if not self._loading:
            self.settingsChanged.emit()

    def _changed(self, key, value, scale):
        slider = self.sliders[key]
        blocked = slider.blockSignals(True)
        slider.setValue(round(value * scale))
        slider.blockSignals(blocked)
        if not self._loading:
            self.presets.setCurrentIndex(0)
            self.settingsChanged.emit()

    def settings(self):
        return CreativeHDRSettings(enabled=self.isChecked(), **{
            name: control.value() for name, control in self.controls.items()
        })

    def _load(self, settings):
        self._loading = True
        try:
            values = asdict(settings)
            for name, control in self.controls.items():
                control.setValue(values[name])
            self.setChecked(settings.enabled)
        finally:
            self._loading = False
        self.settingsChanged.emit()

    def reset_neutral(self):
        self._load(CreativeHDRSettings())
        self.presets.setCurrentIndex(0)

    def apply_preset(self, name):
        presets = {
            'Gentle 400 nits': CreativeHDRSettings(enabled=True, peak_nits=400, strength=.35),
            'Creative 600 nits': CreativeHDRSettings(enabled=True),
            'Bright 1000 nits': CreativeHDRSettings(enabled=True, peak_nits=1000, strength=.6),
        }
        if name in presets:
            self._load(presets[name])
            self.presets.setCurrentText(name)
