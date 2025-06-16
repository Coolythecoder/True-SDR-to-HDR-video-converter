import sys
import os
import subprocess
import signal
import platform
import shutil
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui

class ConversionThread(QtCore.QThread):
    update_status = QtCore.pyqtSignal(str)
    conversion_finished = QtCore.pyqtSignal()

    def __init__(self, ffmpeg_cmd, parent=None):
        super().__init__(parent)
        self.ffmpeg_cmd = ffmpeg_cmd
        self._process = None
        self._cancel_requested = False

    def run(self):
        try:
            self.update_status.emit("Status: Conversion in progress...")
            self._process = subprocess.Popen(
                self.ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self._process.wait()
            if self._cancel_requested:
                self.update_status.emit("Status: Conversion canceled.")
            else:
                self.update_status.emit("Status: Conversion complete.")
        except Exception as e:
            self.update_status.emit(f"Status: Error - {str(e)}")
        finally:
            self.conversion_finished.emit()

    def cancel(self):
        self._cancel_requested = True
        if self._process:
            try:
                if platform.system() == "Windows":
                    self._process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    self._process.terminate()
            except Exception:
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
        layout.addWidget(self.color_convert)
        layout.addWidget(self.embed_metadata)
        layout.addWidget(self.generate_metadata)
        layout.addWidget(self.override_metadata)
        layout.addWidget(self.max_cll_input)
        layout.addWidget(self.max_fall_input)
        layout.addWidget(self.batch_mode)
        layout.addWidget(self.preview_mode)

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
        layout.addWidget(self.start_btn)
        layout.addWidget(self.cancel_btn)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            valid = any(url.toLocalFile().lower().endswith(('.mp4', '.mov', '.mkv')) or os.path.isdir(url.toLocalFile()) for url in event.mimeData().urls())
            if valid:
                event.acceptProposedAction()
                self.setStyleSheet("border: 2px dashed green;")
            else:
                event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet(self.default_style)

    def dropEvent(self, event):
        self.setStyleSheet(self.default_style)
        urls = event.mimeData().urls()
        paths = [url.toLocalFile() for url in urls if url.toLocalFile().lower().endswith(('.mp4', '.mov', '.mkv')) or os.path.isdir(url.toLocalFile())]
        if paths:
            self.input_path.setText(';'.join(paths))

    def select_input_file(self):
        if self.batch_mode.isChecked():
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder")
        else:
            paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Select SDR Videos", "", "Video Files (*.mp4 *.mov *.mkv)")
            path = ';'.join(paths)
        if path:
            self.input_path.setText(path)

    def select_output_file(self):
        if self.batch_mode.isChecked():
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Folder")
        else:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save HDR Video", "", "MP4 Files (*.mp4)")
        if path:
            self.output_path.setText(path)

    def apply_tone_mapping(self, frame, mode):
        if mode == "linear":
            factor = self.linear_scale_input.value()
            return np.clip(frame * factor, 0, 255)
        elif mode == "log":
            factor = self.log_factor_input.value()
            return np.clip(255 * np.log1p(frame * factor) / np.log1p(255 * factor), 0, 255)
        elif mode == "pq":
            gamma = self.pq_gamma_input.value()
            return np.clip(255 * (frame / 255) ** gamma, 0, 255)
        return frame

    def estimate_hdr_metadata(self, video_path):
        self.metadata_thread = MetadataThread(video_path)
        self.metadata_thread.metadata_ready.connect(self.handle_metadata_result)
        self.metadata_thread.start()

    def handle_metadata_result(self, metadata):
        print(f"Estimated Metadata: {metadata}")

    def start_conversion(self):
        input_path = self.input_path.text().strip()
        output_path = self.output_path.text().strip()
        if not input_path or not output_path:
            self.status_label.setText("Status: Please specify input and output paths.")
            return

        crf_value = self.cq_slider.value()
        tone_map_mode = self.tone_map.currentText().lower()
        bit_depth = self.bit_depth.currentText()

        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-vf", f"scale=1920:1080",
            "-c:v", "libx264",
            "-crf", str(crf_value),
            "-pix_fmt", "yuv420p10le" if bit_depth == "10" else "yuv420p",
            output_path
        ]

        self.conversion_thread = ConversionThread(ffmpeg_cmd)
        self.conversion_thread.update_status.connect(self.status_label.setText)
        self.conversion_thread.conversion_finished.connect(self.on_conversion_finished)
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

def main():
    app = QtWidgets.QApplication(sys.argv)
    converter = SDRtoHDRConverter()
    converter.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
