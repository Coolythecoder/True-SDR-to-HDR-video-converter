import sys
import os
import subprocess
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from conversion_paths import find_ffmpeg, plan_conversion_jobs
from hdr_processing import (
    DEFAULT_SDR_PROFILE,
    build_sdr_to_pq_filter,
    find_ffprobe,
    has_zscale,
    probe_sdr_profile,
    x265_sdr_mastering_params,
)
from youtube_hdr_export import (
    YOUTUBE_SUFFIX,
    YOUTUBE_CRF,
    build_youtube_hdr_command,
    supports_libx265,
    validate_youtube_source,
    verify_youtube_hdr_file,
)

class ConversionThread(QtCore.QThread):
    update_status = QtCore.pyqtSignal(str)
    log_output = QtCore.pyqtSignal(str)

    def __init__(self, ffmpeg_commands, parent=None, verify_output=None):
        super().__init__(parent)
        self.ffmpeg_commands = ffmpeg_commands
        self.verify_output = verify_output
        self._process = None
        self._cancel_requested = False

    def run(self):
        succeeded = 0
        failed = 0
        try:
            total = len(self.ffmpeg_commands)
            for number, ffmpeg_cmd in enumerate(self.ffmpeg_commands, 1):
                if self._cancel_requested:
                    break

                name = os.path.basename(ffmpeg_cmd[-1])
                self.update_status.emit(
                    "Status: Converting {}/{}: {}".format(number, total, name)
                )
                self.log_output.emit("Starting {}/{}: {}".format(number, total, name))
                return_code = None
                try:
                    self._process = subprocess.Popen(
                        ffmpeg_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        universal_newlines=True,
                        bufsize=1,
                    )
                    # Cancellation may arrive while Popen is starting.
                    if self._cancel_requested:
                        self._process.terminate()
                    if self._process.stdout:
                        for line in self._process.stdout:
                            self.log_output.emit(line.rstrip())
                        self._process.stdout.close()
                    return_code = self._process.wait()
                except OSError as exc:
                    self.log_output.emit("FFmpeg could not start for {}: {}".format(name, exc))
                finally:
                    self._process = None

                if self._cancel_requested:
                    break
                verification_failed = False
                if return_code == 0 and self.verify_output is not None:
                    try:
                        self.verify_output(ffmpeg_cmd[-1])
                        self.log_output.emit(
                            "Verified YouTube HDR output: {}".format(name)
                        )
                    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                        verification_failed = True
                        self.log_output.emit(
                            "YouTube HDR verification failed for {}: {}".format(name, exc)
                        )
                if return_code == 0 and not verification_failed:
                    succeeded += 1
                else:
                    failed += 1
                    if verification_failed:
                        self.log_output.emit(
                            "{} was encoded but is NOT verified YouTube HDR-ready.".format(name)
                        )
                    else:
                        self.log_output.emit(
                            "Conversion failed for {} (FFmpeg exit code {}).".format(
                                name, return_code if return_code is not None else "unavailable"
                            )
                        )

            if self._cancel_requested:
                self.update_status.emit(
                    "Status: Canceled ({} succeeded, {} failed).".format(succeeded, failed)
                )
            elif failed:
                self.update_status.emit(
                    "Status: Finished with errors ({} succeeded, {} failed).".format(
                        succeeded, failed
                    )
                )
            else:
                self.update_status.emit(
                    "Status: Conversion complete ({} succeeded).".format(succeeded)
                )
        except Exception as exc:
            self.update_status.emit("Status: Unexpected conversion error - {}".format(exc))

    def cancel(self):
        self._cancel_requested = True
        if self._process:
            try:
                self._process.terminate()
            except OSError:
                pass


class MetadataThread(QtCore.QThread):
    metadata_ready = QtCore.pyqtSignal(str)

    def __init__(self, video_path, parent=None):
        super().__init__(parent)
        self.video_path = video_path

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        max_luminance = 0
        frame_luminances = []
        frame_count = 0
        while frame_count < 30 and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            max_luminance = max(max_luminance, gray.max())
            frame_luminances.append(gray.mean())
            frame_count += 1
        cap.release()
        max_cll = int(max_luminance)
        max_fall = int(max(frame_luminances))
        metadata = f"max-cll={max_cll},{max_fall}:max-fall={max_fall}"
        self.metadata_ready.emit(metadata)

class SDRtoHDRConverter(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SDR to HDR Video Converter")
        self.setGeometry(100, 100, 600, 400)
        self.setAcceptDrops(True)
        self.default_style = self.styleSheet()
        self.conversion_thread = None
        self.metadata_thread = None
        self.init_ui()

    def update_hdr_controls(self, depth):
        is_hdr = depth == "10"
        self.hdr_reference_nits.setEnabled(is_hdr)
        self.embed_metadata.setEnabled(is_hdr)
        self.generate_metadata.setEnabled(is_hdr)
        self.override_metadata.setEnabled(is_hdr)
        self.color_processing_label.setText(
            "10-bit HDR: BT.709 SDR to BT.2020 PQ (automatic)"
            if is_hdr else "8-bit output: SDR only; no HDR metadata"
        )
        self.toggle_override_fields()

    def toggle_override_fields(self):
        enabled = self.bit_depth.currentText() == "10" and self.override_metadata.isChecked()
        self.max_cll_input.setEnabled(enabled)
        self.max_fall_input.setEnabled(enabled)

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()
        self.input_btn = QtWidgets.QPushButton("Select SDR Video")
        self.input_btn.clicked.connect(self.select_input_file)
        self.input_path = QtWidgets.QLineEdit()
        layout.addWidget(self.input_btn)
        layout.addWidget(self.input_path)

        self.output_btn = QtWidgets.QPushButton("Select Output Location")
        self.output_btn.clicked.connect(self.select_output_file)
        self.output_path = QtWidgets.QLineEdit()
        layout.addWidget(self.output_btn)
        layout.addWidget(self.output_path)

        self.hdr_reference_label = QtWidgets.QLabel("HDR Reference White (nits):")
        self.hdr_reference_nits = QtWidgets.QSpinBox()
        self.hdr_reference_nits.setRange(80, 400)
        self.hdr_reference_nits.setValue(200)
        self.hdr_reference_nits.setToolTip(
            "SDR 100-nit diffuse white maps to this PQ luminance. "
            "200 nits follows MovieLabs BT.709-to-HDR10 guidance."
        )
        layout.addWidget(self.hdr_reference_label)
        layout.addWidget(self.hdr_reference_nits)

        self.bit_depth_label = QtWidgets.QLabel("Output Bit Depth:")
        self.bit_depth = QtWidgets.QComboBox()
        self.bit_depth.addItems(["10", "8"])
        self.bit_depth.setToolTip("10-bit output is HDR PQ; 8-bit output stays SDR.")
        layout.addWidget(self.bit_depth_label)
        layout.addWidget(self.bit_depth)

        self.color_processing_label = QtWidgets.QLabel()
        self.embed_metadata = QtWidgets.QCheckBox("Embed HDR10 mastering metadata (x265)")
        self.embed_metadata.setChecked(True)
        self.generate_metadata = QtWidgets.QCheckBox("Generate Static HDR Metadata")
        self.override_metadata = QtWidgets.QCheckBox("Override HDR Metadata")
        self.max_cll_input = QtWidgets.QLineEdit()
        self.max_fall_input = QtWidgets.QLineEdit()
        self.max_cll_input.setPlaceholderText("MaxCLL")
        self.max_fall_input.setPlaceholderText("MaxFALL")
        self.override_metadata.stateChanged.connect(self.toggle_override_fields)
        self.batch_mode = QtWidgets.QCheckBox("Batch Convert (folder)")
        self.preview_mode = QtWidgets.QCheckBox("Real-time Preview (not implemented)")
        self.preview_mode.setEnabled(False)
        self.preview_mode.setToolTip(
            "The app does not yet contain an HDR preview. "
            "Check the finished clip in an HDR-capable player."
        )
        self.gpu_checkbox = QtWidgets.QCheckBox("Use GPU Acceleration (NVENC)")
        self.audio_passthrough = QtWidgets.QCheckBox("Enable Audio Passthrough")
        layout.addWidget(self.audio_passthrough)
        layout.addWidget(self.color_processing_label)
        layout.addWidget(self.embed_metadata)
        layout.addWidget(self.generate_metadata)
        layout.addWidget(self.override_metadata)
        layout.addWidget(self.max_cll_input)
        layout.addWidget(self.max_fall_input)
        layout.addWidget(self.batch_mode)
        layout.addWidget(self.preview_mode)
        layout.addWidget(self.gpu_checkbox)

        self.cq_label = QtWidgets.QLabel("Constant Quality (CRF):")
        self.cq_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.cq_slider.setMinimum(0)
        self.cq_slider.setMaximum(51)
        self.cq_slider.setValue(23)
        self.cq_slider.setTickInterval(1)
        self.cq_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.cq_value_label = QtWidgets.QLabel("23")
        self.cq_slider.valueChanged.connect(lambda val: self.cq_value_label.setText(str(val)))
        layout.addWidget(self.cq_label)
        layout.addWidget(self.cq_slider)
        layout.addWidget(self.cq_value_label)

        self.start_btn = QtWidgets.QPushButton("Start Conversion")
        self.start_btn.clicked.connect(self.start_conversion)
        self.youtube_btn = QtWidgets.QPushButton("Export YouTube HDR MP4")
        self.youtube_btn.clicked.connect(self.start_youtube_export)
        self.youtube_btn.setToolTip(
            "YouTube upload preset: HEVC Main 10 / PQ BT.2020 / static HDR10 "
            "metadata / 48 kHz AAC stereo / MP4 faststart. "
            "Uses CPU x265 for complete HDR10 metadata."
        )
        self.cancel_btn = QtWidgets.QPushButton("Cancel Conversion")
        self.cancel_btn.clicked.connect(self.cancel_conversion)
        self.status_label = QtWidgets.QLabel("Status: Waiting...")
        self.log_output = QtWidgets.QPlainTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(QtWidgets.QLabel("Conversion Log:"))
        layout.addWidget(self.log_output)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.youtube_btn)
        layout.addWidget(QtWidgets.QLabel(
            "YouTube HDR: 10-bit HEVC (CPU x265), CRF 16, AAC 48 kHz stereo. "
            "Output is checked after encoding."
        ))
        layout.addWidget(self.cancel_btn)
        layout.addWidget(self.status_label)
        self.setLayout(layout)
        self.bit_depth.currentTextChanged.connect(self.update_hdr_controls)
        self.update_hdr_controls(self.bit_depth.currentText())

    def select_input_file(self):
        if self.batch_mode.isChecked():
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Input Folder")
        else:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select SDR Video", "", "Video Files (*.mp4 *.mov *.mkv)"
            )
        if path:
            self.input_path.setText(path)

    def select_output_file(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self.output_path.setText(path)

    def build_ffmpeg_command(
        self, ffmpeg_exe, input_path, output_path, use_gpu, source_profile=None
    ):
        bit_depth = self.bit_depth.currentText()
        quality = str(self.cq_slider.value())
        source_profile = source_profile or DEFAULT_SDR_PROFILE

        if use_gpu:
            encoder = "hevc_nvenc" if bit_depth == "10" else "h264_nvenc"
            encoder_settings = ["-rc", "constqp", "-qp", quality]
        else:
            encoder = "libx265" if bit_depth == "10" else "libx264"
            encoder_settings = ["-crf", quality]

        pixel_format = (
            "p010le" if use_gpu else "yuv420p10le"
        ) if bit_depth == "10" else "yuv420p"

        ffmpeg_cmd = [ffmpeg_exe, "-y", "-i", input_path]
        if bit_depth == "10":
            # IMPORTANT: Do not tag SDR YUV as PQ. Actually transform the
            # pixel values, through display-linear BT.2020, before encoding.
            ffmpeg_cmd += [
                "-vf",
                build_sdr_to_pq_filter(
                    source_profile, self.hdr_reference_nits.value(), pixel_format
                ),
                "-color_primaries", "bt2020",
                "-color_trc", "smpte2084",
                "-colorspace", "bt2020nc",
                "-color_range", "tv",
            ]

        ffmpeg_cmd += [
            "-c:v", encoder,
            "-pix_fmt", pixel_format,
        ] + encoder_settings

        # Static ST 2086 + MaxCLL/MaxFALL SEI in x265 only. NVENC still
        # receives valid PQ/BT.2020 signalling, but not invented ST 2086.
        if (
            bit_depth == "10" and encoder == "libx265"
            and source_profile["primaries"] == "bt709"
            and (self.embed_metadata.isChecked() or self.generate_metadata.isChecked())
        ):
            max_cll = self.hdr_reference_nits.value()
            max_fall = 0  # "unknown", not an estimated per-frame average
            if self.override_metadata.isChecked():
                cll_text = self.max_cll_input.text().strip()
                fall_text = self.max_fall_input.text().strip()
                if cll_text:
                    if not cll_text.isdigit():
                        raise ValueError("MaxCLL must be a positive integer.")
                    max_cll = int(cll_text)
                if fall_text:
                    if not fall_text.isdigit():
                        raise ValueError("MaxFALL must be a nonnegative integer.")
                    max_fall = int(fall_text)

            ffmpeg_cmd += [
                "-x265-params",
                x265_sdr_mastering_params(
                    self.hdr_reference_nits.value(), max_cll, max_fall
                ),
            ]

        if self.audio_passthrough.isChecked():
            ffmpeg_cmd += ["-c:a", "copy"]
        else:
            ffmpeg_cmd += ["-c:a", "aac", "-b:a", "128k"]
        ffmpeg_cmd.append(output_path)
        return ffmpeg_cmd

    def start_conversion(self):
        self._start_conversion(youtube_mode=False)

    def start_youtube_export(self):
        self._start_conversion(youtube_mode=True)

    def _start_conversion(self, youtube_mode=False):
        if self.conversion_thread and self.conversion_thread.isRunning():
            self.status_label.setText("Status: Conversion already in progress.")
            return

        input_path = self.input_path.text().strip()
        output_path = self.output_path.text().strip()
        if not input_path or not output_path:
            self.status_label.setText("Status: Please specify input and output paths.")
            return

        bit_depth = "10" if youtube_mode else self.bit_depth.currentText()
        suffix = YOUTUBE_SUFFIX if youtube_mode else (
            "_HDR" if bit_depth == "10" else "_SDR"
        )
        try:
            jobs = plan_conversion_jobs(
                input_path, output_path, self.batch_mode.isChecked(),
                output_suffix=suffix,
            )
        except (ValueError, OSError) as exc:
            self.status_label.setText("Status: {}".format(exc))
            return

        ffmpeg_exe = find_ffmpeg()
        if not ffmpeg_exe:
            self.status_label.setText(
                "Status: FFmpeg not found. Install FFmpeg and add it to PATH."
            )
            return
        if bit_depth == "10" and not has_zscale(ffmpeg_exe):
            self.status_label.setText(
                "Status: FFmpeg lacks the zscale/zimg colour filter. "
                "Install a full FFmpeg build before exporting HDR."
            )
            return

        ffprobe_exe = find_ffprobe(ffmpeg_exe)
        if youtube_mode:
            if not ffprobe_exe:
                self.status_label.setText(
                    "Status: YouTube HDR requires FFprobe for input/output verification."
                )
                return
            if not supports_libx265(ffmpeg_exe):
                self.status_label.setText(
                    "Status: YouTube HDR requires an FFmpeg build with libx265."
                )
                return
        elif not ffprobe_exe:
            self.log_output.appendPlainText(
                "Warning: FFprobe not found. Assuming SDR BT.709 limited-range source."
            )

        use_gpu = False if youtube_mode else self.gpu_checkbox.isChecked()
        if use_gpu:
            try:
                result = subprocess.run(
                    [ffmpeg_exe, "-hide_banner", "-encoders"],
                    capture_output=True, text=True, timeout=15,
                )
                available_encoders = result.stdout if result.returncode == 0 else ""
            except (OSError, subprocess.TimeoutExpired):
                available_encoders = ""

            required_encoder = (
                "hevc_nvenc" if bit_depth == "10" else "h264_nvenc"
            )
            if required_encoder not in available_encoders:
                self.log_output.appendPlainText(
                    "GPU encoder {} unavailable; falling back to CPU.".format(
                        required_encoder
                    )
                )
                use_gpu = False

        if bit_depth == "10" and use_gpu and (
            self.embed_metadata.isChecked() or self.generate_metadata.isChecked()
        ):
            self.log_output.appendPlainText(
                "Notice: NVENC produces PQ/BT.2020 HDR, but this path does not "
                "embed x265 mastering-display SEI. Select CPU/x265 for ST 2086."
            )

        commands = []
        try:
            for source, destination in jobs:
                profile = probe_sdr_profile(ffprobe_exe, source)
                if youtube_mode:
                    validate_youtube_source(ffprobe_exe, source, profile)
                    white_nits = self.hdr_reference_nits.value()
                    max_cll, max_fall = white_nits, 0
                    if self.override_metadata.isChecked():
                        text_cll = self.max_cll_input.text().strip()
                        text_fall = self.max_fall_input.text().strip()
                        if text_cll:
                            if not text_cll.isdigit():
                                raise ValueError("MaxCLL must be a positive integer.")
                            max_cll = int(text_cll)
                        if text_fall:
                            if not text_fall.isdigit():
                                raise ValueError("MaxFALL must be a nonnegative integer.")
                            max_fall = int(text_fall)
                    commands.append(build_youtube_hdr_command(
                        ffmpeg_exe, source, destination, profile,
                        white_nits=white_nits, crf=YOUTUBE_CRF,
                        max_cll=max_cll, max_fall=max_fall,
                    ))
                else:
                    if bit_depth == "10" and profile["primaries"] != "bt709":
                        self.log_output.appendPlainText(
                            "Notice: {} has non-BT.709 primaries; static mastering-display "
                            "metadata will be omitted.".format(os.path.basename(source))
                        )
                    commands.append(self.build_ffmpeg_command(
                        ffmpeg_exe, source, destination, use_gpu, profile
                    ))
        except ValueError as exc:
            self.status_label.setText("Status: {}".format(exc))
            return

        verify_output = None
        if youtube_mode:
            self.log_output.appendPlainText(
                "YouTube HDR preset: CPU x265 Main 10, BT.2020/PQ, "
                "HDR10 mastering metadata, 48 kHz AAC stereo, MP4 faststart. "
                "Normal quality/GPU/audio selections are not used."
            )
            verify_output = (
                lambda output: verify_youtube_hdr_file(ffprobe_exe, output)
            )

        self.conversion_thread = ConversionThread(
            commands, self, verify_output=verify_output
        )
        self.conversion_thread.update_status.connect(self.status_label.setText)
        self.conversion_thread.log_output.connect(self.log_output.appendPlainText)
        self.conversion_thread.finished.connect(self.on_conversion_finished)
        self.start_btn.setEnabled(False)
        self.youtube_btn.setEnabled(False)
        self.conversion_thread.start()

    def cancel_conversion(self):
        reply = QtWidgets.QMessageBox.question(self, 'Cancel Conversion',
                                               'Are you sure you want to cancel the conversion?',
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                               QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            if self.conversion_thread and self.conversion_thread.isRunning():
                self.conversion_thread.cancel()
                self.status_label.setText("Status: Cancel requested...")

    def on_conversion_finished(self):
        self.conversion_thread = None
        self.start_btn.setEnabled(True)
        self.youtube_btn.setEnabled(True)

def main():
    app = QtWidgets.QApplication(sys.argv)
    converter = SDRtoHDRConverter()
    converter.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
