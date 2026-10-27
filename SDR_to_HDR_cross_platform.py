import sys
import os
import subprocess
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from conversion_paths import find_ffmpeg, plan_conversion_jobs

class ConversionThread(QtCore.QThread):
    update_status = QtCore.pyqtSignal(str)
    log_output = QtCore.pyqtSignal(str)

    def __init__(self, ffmpeg_commands, parent=None):
        super().__init__(parent)
        self.ffmpeg_commands = ffmpeg_commands
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
                if return_code == 0:
                    succeeded += 1
                else:
                    failed += 1
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

    def update_tone_controls(self, text):
        is_linear = text.lower() == "linear"
        is_pq = text.lower() == "pq"
        is_log = text.lower() == "log"

        self.linear_scale_label.setVisible(is_linear)
        self.linear_scale_input.setVisible(is_linear)
        self.pq_gamma_label.setVisible(is_pq)
        self.pq_gamma_input.setVisible(is_pq)
        self.log_factor_label.setVisible(is_log)
        self.log_factor_input.setVisible(is_log)

    def toggle_override_fields(self):
        enabled = self.override_metadata.isChecked()
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

        self.tone_map_label = QtWidgets.QLabel("Tone Mapping Type:")
        self.tone_map = QtWidgets.QComboBox()
        self.tone_map.addItems(["Linear", "Log", "PQ"])
        layout.addWidget(self.tone_map_label)
        layout.addWidget(self.tone_map)

        self.linear_scale_label = QtWidgets.QLabel("Linear Scale Factor:")
        self.linear_scale_input = QtWidgets.QDoubleSpinBox()
        self.linear_scale_input.setRange(0.1, 5.0)
        self.linear_scale_input.setValue(1.5)
        layout.addWidget(self.linear_scale_label)
        layout.addWidget(self.linear_scale_input)

        self.pq_gamma_label = QtWidgets.QLabel("PQ Gamma Exponent:")
        self.pq_gamma_input = QtWidgets.QDoubleSpinBox()
        self.pq_gamma_input.setRange(0.1, 5.0)
        self.pq_gamma_input.setValue(2.2)
        layout.addWidget(self.pq_gamma_label)
        layout.addWidget(self.pq_gamma_input)

        self.log_factor_label = QtWidgets.QLabel("Log Compression Factor:")
        self.log_factor_input = QtWidgets.QDoubleSpinBox()
        self.log_factor_input.setRange(0.1, 10.0)
        self.log_factor_input.setValue(1.0)
        layout.addWidget(self.log_factor_label)
        layout.addWidget(self.log_factor_input)

        self.tone_map.currentTextChanged.connect(self.update_tone_controls)
        self.update_tone_controls(self.tone_map.currentText())

        self.bit_depth_label = QtWidgets.QLabel("Output Bit Depth:")
        self.bit_depth = QtWidgets.QComboBox()
        self.bit_depth.addItems(["8", "10"])
        layout.addWidget(self.bit_depth_label)
        layout.addWidget(self.bit_depth)

        self.color_convert = QtWidgets.QCheckBox("Convert BT.709 to BT.2020")
        self.embed_metadata = QtWidgets.QCheckBox("Embed HDR10 Metadata")
        self.generate_metadata = QtWidgets.QCheckBox("Generate Static HDR Metadata")
        self.override_metadata = QtWidgets.QCheckBox("Override HDR Metadata")
        self.max_cll_input = QtWidgets.QLineEdit()
        self.max_fall_input = QtWidgets.QLineEdit()
        self.max_cll_input.setPlaceholderText("MaxCLL")
        self.max_fall_input.setPlaceholderText("MaxFALL")
        self.override_metadata.stateChanged.connect(self.toggle_override_fields)
        self.batch_mode = QtWidgets.QCheckBox("Batch Convert (folder)")
        self.preview_mode = QtWidgets.QCheckBox("Real-time Preview")
        self.gpu_checkbox = QtWidgets.QCheckBox("Use GPU Acceleration (NVENC)")
        self.audio_passthrough = QtWidgets.QCheckBox("Enable Audio Passthrough")
        layout.addWidget(self.audio_passthrough)
        layout.addWidget(self.color_convert)
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
        self.cancel_btn = QtWidgets.QPushButton("Cancel Conversion")
        self.cancel_btn.clicked.connect(self.cancel_conversion)
        self.status_label = QtWidgets.QLabel("Status: Waiting...")
        self.log_output = QtWidgets.QPlainTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(QtWidgets.QLabel("Conversion Log:"))
        layout.addWidget(self.log_output)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.cancel_btn)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

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

    def build_ffmpeg_command(self, ffmpeg_exe, input_path, output_path, use_gpu):
        bit_depth = self.bit_depth.currentText()
        quality = str(self.cq_slider.value())

        if use_gpu:
            encoder = "hevc_nvenc" if bit_depth == "10" else "h264_nvenc"
            encoder_settings = ["-rc", "constqp", "-qp", quality]
        else:
            encoder = "libx265" if bit_depth == "10" else "libx264"
            encoder_settings = ["-crf", quality]

        pixel_format = (
            "p010le" if use_gpu else "yuv420p10le"
        ) if bit_depth == "10" else "yuv420p"

        ffmpeg_cmd = [
            ffmpeg_exe, "-y", "-i", input_path,
            "-c:v", encoder,
            "-pix_fmt", pixel_format,
        ] + encoder_settings

        if self.embed_metadata.isChecked() or self.generate_metadata.isChecked():
            ffmpeg_cmd += [
                "-color_primaries", "bt2020",
                "-color_trc", "smpte2084",
                "-colorspace", "bt2020nc",
            ]
            # x265 parameters are incompatible with hevc_nvenc.
            # Accurate SDR-to-PQ colour processing is a separate concern.
            if encoder == "libx265":
                master_display = (
                    "G(13250,34500)B(7500,3000)R(34000,16000)"
                    "WP(15635,16450)L(10000000,1)"
                )
                max_cll, max_fall = "1000", "400"
                if self.override_metadata.isChecked():
                    cll = self.max_cll_input.text().strip()
                    fall = self.max_fall_input.text().strip()
                    if cll.isdigit():
                        max_cll = cll
                    if fall.isdigit():
                        max_fall = fall
                ffmpeg_cmd += [
                    "-x265-params",
                    "master-display={}:max-cll={},{}".format(master_display, max_cll, max_fall),
                ]

        if self.audio_passthrough.isChecked():
            ffmpeg_cmd += ["-c:a", "copy"]
        else:
            ffmpeg_cmd += ["-c:a", "aac", "-b:a", "128k"]
        ffmpeg_cmd.append(output_path)
        return ffmpeg_cmd

    def start_conversion(self):
        if self.conversion_thread and self.conversion_thread.isRunning():
            self.status_label.setText("Status: Conversion already in progress.")
            return

        input_path = self.input_path.text().strip()
        output_path = self.output_path.text().strip()
        if not input_path or not output_path:
            self.status_label.setText("Status: Please specify input and output paths.")
            return

        try:
            jobs = plan_conversion_jobs(
                input_path, output_path, self.batch_mode.isChecked()
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

        use_gpu = self.gpu_checkbox.isChecked()
        if use_gpu:
            try:
                result = subprocess.run(
                    [ffmpeg_exe, "-hide_banner", "-encoders"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                available_encoders = result.stdout if result.returncode == 0 else ""
            except (OSError, subprocess.TimeoutExpired):
                available_encoders = ""

            required_encoder = (
                "hevc_nvenc" if self.bit_depth.currentText() == "10"
                else "h264_nvenc"
            )
            if required_encoder not in available_encoders:
                self.log_output.appendPlainText(
                    "GPU encoder {} unavailable; falling back to CPU.".format(
                        required_encoder
                    )
                )
                use_gpu = False

        commands = [
            self.build_ffmpeg_command(ffmpeg_exe, source, destination, use_gpu)
            for source, destination in jobs
        ]
        self.conversion_thread = ConversionThread(commands, self)
        self.conversion_thread.update_status.connect(self.status_label.setText)
        self.conversion_thread.log_output.connect(self.log_output.appendPlainText)
        self.conversion_thread.finished.connect(self.on_conversion_finished)
        self.start_btn.setEnabled(False)
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

def main():
    app = QtWidgets.QApplication(sys.argv)
    converter = SDRtoHDRConverter()
    converter.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
