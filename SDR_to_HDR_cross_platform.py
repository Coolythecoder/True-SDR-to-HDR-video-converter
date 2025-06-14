import sys
import os
import subprocess
import signal
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import platform
import shutil


class SDRtoHDRConverter(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SDR to HDR Video Converter")
        self.setGeometry(100, 100, 600, 400)
        self.setAcceptDrops(True)
        self.default_style = self.styleSheet()
        self.cancel_requested = False
        self.current_process = None
        self.init_ui()

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

        self.start_btn = QtWidgets.QPushButton("Start Conversion")
        self.start_btn.clicked.connect(self.start_conversion)
        self.cancel_btn = QtWidgets.QPushButton("Cancel Conversion")
        self.cancel_btn.clicked.connect(self.cancel_conversion)
        self.status_label = QtWidgets.QLabel("Status: Waiting...")
        layout.addWidget(self.start_btn)
        layout.addWidget(self.cancel_btn)
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def cancel_conversion(self):
        reply = QtWidgets.QMessageBox.question(
            self,
            'Cancel Conversion',
            'Are you sure you want to cancel the conversion?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self.cancel_requested = True
            if self.current_process:
                if platform.system() == "Windows":
                    self.current_process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    self.current_process.terminate()
            self.status_label.setText("Status: Cancel requested...")

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
            return np.clip(frame * 1.5, 0, 255)
        elif mode == "log":
            return np.clip(255 * np.log1p(frame) / np.log1p(255), 0, 255)
        elif mode == "pq":
            return np.clip(255 * (frame / 255) ** 2.2, 0, 255)
        return frame

    def estimate_hdr_metadata(self, video_path):
        cap = cv2.VideoCapture(video_path)
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
        return f"max-cll={max_cll},{max_fall}:max-fall={max_fall}"

    def start_conversion(self):
        if not shutil.which("ffmpeg"):
            self.status_label.setText("Status: ffmpeg not found in PATH.")
            QtWidgets.QMessageBox.critical(self, "ffmpeg Not Found",
                                           "ffmpeg executable is not found in your system PATH. Please install ffmpeg.")
            return

        self.cancel_requested = False
        self.current_process = None
        input_data = self.input_path.text()
        output_file = self.output_path.text()
        tone = self.tone_map.currentText().lower()
        bit_depth = self.bit_depth.currentText()
        convert_color = self.color_convert.isChecked()
        embed_meta = self.embed_metadata.isChecked()
        batch = self.batch_mode.isChecked()
        preview = self.preview_mode.isChecked()

        if not input_data:
            self.status_label.setText("Status: No input selected.")
            return
        if not output_file:
            self.status_label.setText("Status: No output path selected.")
            return

        self.status_label.setText("Status: Converting...")

        input_paths = input_data.split(';')
        files = []
        for path in input_paths:
            if os.path.isdir(path):
                files.extend([os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.mp4', '.mov', '.mkv'))])
            elif os.path.isfile(path):
                files.append(path)

        for f in files:
            if self.cancel_requested:
                self.status_label.setText("Status: Conversion cancelled.")
                return

            out = os.path.join(output_file, os.path.basename(f)) if batch else output_file

            if preview:
                cap = cv2.VideoCapture(f)
                while cap.isOpened():
                    if self.cancel_requested:
                        cap.release()
                        cv2.destroyAllWindows()
                        self.status_label.setText("Status: Conversion cancelled.")
                        return
                    ret, frame = cap.read()
                    if not ret:
                        break
                    mapped = self.apply_tone_mapping(frame.astype(np.float32), tone)
                    cv2.imshow("Preview", mapped.astype(np.uint8))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()
                continue

            cmd = [
                "ffmpeg", "-i", f,
                "-c:v", "libx265", "-preset", "slow", "-crf", "18",
                "-color_primaries", "bt2020" if convert_color else "bt709",
                "-color_trc", "smpte2084" if tone == "pq" else "bt709",
                "-colorspace", "bt2020nc" if convert_color else "bt709"
            ]

            if embed_meta:
                if self.generate_metadata.isChecked():
                    if self.override_metadata.isChecked():
                        cll = self.max_cll_input.text() or "1000"
                        fall = self.max_fall_input.text() or "400"
                        dynamic_metadata = f"max-cll={cll},{fall}:max-fall={fall}"
                    else:
                        dynamic_metadata = self.estimate_hdr_metadata(f)
                    static_metadata = (
                        "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:"
                        "master-display=G(13250,34500)B(7500,3000)R(34000,16000)"
                        "WP(15635,16450)L(10000000,1)"
                    )
                    metadata = f"hdr-opt=1:{static_metadata}:{dynamic_metadata}"
                    cmd += ["-x265-params", metadata]
                else:
                    cmd += ["-x265-params", "hdr-opt=1"]

            cmd += [out]

            try:
                self.current_process = subprocess.Popen(cmd)
                self.current_process.wait()
                if self.cancel_requested:
                    self.status_label.setText("Status: Conversion cancelled.")
                    return
            except subprocess.CalledProcessError:
                self.status_label.setText(f"Status: Failed on {f}")
                return

        self.status_label.setText("Status: Conversion complete.")

def main():
    app = QtWidgets.QApplication(sys.argv)
    converter = SDRtoHDRConverter()
    converter.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
