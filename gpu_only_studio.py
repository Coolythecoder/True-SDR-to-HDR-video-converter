"""HDR Video Studio with an opt-in, strictly GPU-resident custom export.

Run this entry point for the optional CUDA mode. The original entry point and
YouTube builder remain unchanged. CUDA is loaded only inside the export worker.
"""
import sys
from PyQt5 import QtWidgets
from SDR_to_HDR_cross_platform import SDRtoHDRConverter, ConversionThread
from conversion_paths import find_ffmpeg, plan_conversion_jobs
from hdr_processing import find_ffprobe
from workspace_ui import set_exporting
from gpu_only_export import GPUOnlySettings, export_gpu_only
from youtube_export_analysis import ExportCancelled


class GPUOnlyThread(ConversionThread):
    def __init__(self, jobs, ffmpeg, ffprobe, settings, parent=None):
        super().__init__([], parent)
        self.jobs, self.ffmpeg, self.ffprobe, self.settings = jobs, ffmpeg, ffprobe, settings

    def run(self):
        succeeded, failed = 0, 0
        def register(process):
            self._process = process
            if process is not None and self._cancel_requested:
                process.terminate()
        for i, (source, destination) in enumerate(self.jobs, 1):
            if self._cancel_requested:
                break
            self.update_status.emit('Status: GPU-only export {}/{}'.format(i, len(self.jobs)))
            self.log_output.emit('Starting GPU-only {}/{}: {}'.format(i, len(self.jobs), destination))
            try:
                export_gpu_only(self.ffmpeg, self.ffprobe, source, destination, self.settings,
                                log=self.log_output.emit, cancelled=lambda: self._cancel_requested,
                                register=register)
                succeeded += 1
            except ExportCancelled:
                self._cancel_requested = True
                break
            except Exception as exc:
                if self._cancel_requested:
                    break
                failed += 1
                self.log_output.emit('GPU-only export stopped: {}. Existing output was not replaced; '
                                     'CPU fallback is disabled.'.format(exc))
            finally:
                self._process = None
        state = 'Canceled' if self._cancel_requested else ('Finished with errors' if failed else 'Conversion complete')
        self.update_status.emit('Status: {} ({} succeeded, {} failed).'.format(state, succeeded, failed))


class GPUOnlyStudio(SDRtoHDRConverter):
    def init_ui(self):
        super().init_ui()
        self.setWindowTitle('HDR Video Studio - optional CUDA GPU-only export')
        custom = next(g for g in self.export_page.findChildren(QtWidgets.QGroupBox) if g.title() == 'Custom export')
        self.gpu_only_checkbox = QtWidgets.QCheckBox('GPU-only video pipeline (NVIDIA CUDA, experimental)')
        self.gpu_only_checkbox.setChecked(False)
        self.gpu_only_checkbox.setToolTip(
            'Custom export only: NVDEC, colour conversion, grading, all-pixel light analysis, '
            'PQ/10-bit conversion and NVENC run on NVIDIA GPU. No silent CPU fallback. '
            'CPU still handles control, audio, file IO and verification. Requires CUDA dependencies. '
            'BT.709 H.264/HEVC 4:2:0 progressive CFR inputs only; output is MP4. '
            'Preview and YouTube settings are unchanged.')
        note = QtWidgets.QLabel('GPU-only is optional and uses a separate native CUDA backend. '
                               'The existing hybrid path remains the default. Audio/mux stay on CPU; '
                               'YouTube and live preview are unaffected.')
        note.setWordWrap(True)
        custom.layout().insertWidget(1, self.gpu_only_checkbox)
        custom.layout().insertWidget(2, note)
        self._prior_gpu_controls = None
        self._normal_quality_label = self.cq_label.text()
        self._normal_gpu_tooltip = self.gpu_checkbox.toolTip()
        self.gpu_only_checkbox.toggled.connect(self._gpu_only_toggled)

    def _gpu_only_toggled(self, enabled):
        if enabled:
            self._prior_gpu_controls = self.gpu_checkbox.isChecked(), self.bit_depth.currentText()
            self.gpu_checkbox.setChecked(True)
            self.bit_depth.setCurrentText('10')
        elif self._prior_gpu_controls:
            gpu, depth = self._prior_gpu_controls
            self.gpu_checkbox.setChecked(gpu)
            self.bit_depth.setCurrentText(depth)
            self._prior_gpu_controls = None
        self.cq_label.setText('Native NVENC constant QP (P4; lower = higher quality)' if enabled else self._normal_quality_label)
        self.gpu_checkbox.setToolTip('Native CUDA mode includes measured HDR10 mastering/content-light SEI.' if enabled else self._normal_gpu_tooltip)
        self.gpu_checkbox.setEnabled(not enabled)
        self.bit_depth.setEnabled(not enabled)

    def _start_conversion(self, youtube_mode=False):
        if youtube_mode or not self.gpu_only_checkbox.isChecked():
            return super()._start_conversion(youtube_mode=youtube_mode)
        if self.conversion_thread and self.conversion_thread.isRunning():
            self.status_label.setText('Status: Conversion already in progress.')
            return
        try:
            settings = GPUOnlySettings(self.creative_panel.settings(), self.hdr_reference_nits.value(),
                                       self.cq_slider.value(), self.audio_passthrough.isChecked()).validate()
            jobs = plan_conversion_jobs(self.input_path.text().strip(), self.output_path.text().strip(),
                                        self.batch_mode.isChecked(), output_suffix='_HDR_GPU')
            ffmpeg = find_ffmpeg()
            ffprobe = find_ffprobe(ffmpeg) if ffmpeg else None
            if not ffmpeg or not ffprobe:
                raise ValueError('GPU-only export needs FFmpeg and FFprobe for audio/mux and validation.')
        except (ValueError, OSError) as exc:
            self.status_label.setText('Status: ' + str(exc))
            return
        self.log_output.appendPlainText(
            'GPU-only custom export selected. The entire clip is analysed on CUDA; measured HDR10 '
            'metadata is embedded. Manual metadata overrides are not used. YouTube settings are unchanged.')
        self.conversion_thread = GPUOnlyThread(jobs, ffmpeg, ffprobe, settings, self)
        self.conversion_thread.update_status.connect(self.status_label.setText)
        self.conversion_thread.log_output.connect(self.log_output.appendPlainText)
        self.conversion_thread.finished.connect(self.on_conversion_finished)
        self.start_btn.setEnabled(False)
        self.youtube_btn.setEnabled(False)
        set_exporting(self, True)
        self.conversion_thread.start()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = GPUOnlyStudio()
    window.show()
    if '--smoke-test' in sys.argv:
        # CI checks the packaged GUI without needing a GPU; never starts export.
        from PyQt5 import QtCore
        QtCore.QTimer.singleShot(1000, app.quit)
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
