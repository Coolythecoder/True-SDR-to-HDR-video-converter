"""Optional grading panel. No worker reads live Qt controls."""
from dataclasses import asdict
from PyQt5 import QtWidgets
from creative_hdr import CreativeHDRSettings


class CreativeHDRPanel(QtWidgets.QGroupBox):
    def __init__(self, parent=None):
        super().__init__('Creative HDR grade (optional)', parent)
        self.setCheckable(True)
        self.setChecked(False)
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(
            'Linear-light BT.2020 grading before PQ. Creative highlight expansion, '
            'not recovery of clipped detail. Content-light metadata is rescanned '
            'after grading. Only 10-bit HDR exports support this mode.'
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.presets = QtWidgets.QComboBox()
        self.presets.addItems(['Custom', 'Gentle 400 nits', 'Creative 600 nits', 'Bright 1000 nits'])
        layout.addWidget(self.presets)
        form = QtWidgets.QFormLayout()
        self.controls = {}
        definitions = [
            ('peak_nits', 'Peak ceiling (nits)', 200, 4000, 50, 0),
            ('strength', 'Highlight expansion strength', 0, 1, .05, 2),
            ('knee', 'Highlight start (linear / reference white)', .5, .95, .05, 2),
            ('rolloff', 'Highlight roll-off softness', .05, 1, .05, 2),
            ('exposure', 'Exposure (stops)', -2, 2, .1, 2),
            ('contrast', 'Contrast (18% grey pivot)', .7, 1.4, .05, 2),
            ('midtones', 'Midtone balance (stops)', -1, 1, .1, 2),
            ('shadows', 'Shadow adjustment (stops)', -2, 2, .1, 2),
            ('black_floor', 'Black lift (nits)', 0, 2, .05, 2),
            ('saturation', 'Saturation', 0, 1.5, .05, 2),
            ('vibrance', 'Vibrance', -.5, 1, .05, 2),
            ('temperature', 'Temperature (cool / warm, not Kelvin)', -1, 1, .05, 2),
            ('tint', 'Tint (green / magenta)', -1, 1, .05, 2),
        ]
        defaults = asdict(CreativeHDRSettings())
        for key, label, low, high, step, decimals in definitions:
            control = QtWidgets.QDoubleSpinBox()
            control.setRange(low, high)
            control.setDecimals(decimals)
            control.setSingleStep(step)
            control.setValue(defaults[key])
            control.valueChanged.connect(self._custom)
            self.controls[key] = control
            form.addRow(label, control)
        layout.addLayout(form)
        reset = QtWidgets.QPushButton('Reset to accurate conversion (disable grading)')
        reset.clicked.connect(self.reset_neutral)
        layout.addWidget(reset)
        self.presets.activated[str].connect(self.apply_preset)

    def _custom(self, _value):
        self.presets.setCurrentIndex(0)

    def settings(self):
        return CreativeHDRSettings(enabled=self.isChecked(), **{
            name: control.value() for name, control in self.controls.items()
        })

    def _load(self, settings):
        values = asdict(settings)
        for name, control in self.controls.items():
            control.setValue(values[name])
        self.setChecked(settings.enabled)

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
