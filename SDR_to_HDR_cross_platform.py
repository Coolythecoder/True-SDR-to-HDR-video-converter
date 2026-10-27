import sys
import os
import subprocess
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from conversion_paths import find_ffmpeg, plan_conversion_jobs
from hdr_processing import (
    DEFAULT_SDR_PROFILE, build_sdr_to_pq_filter, find_ffprobe, has_zscale,
    probe_sdr_profile, x265_sdr_mastering_params,
)
from youtube_hdr_export import (
    YOUTUBE_SUFFIX, YOUTUBE_CRF, YOUTUBE_QUALITY_PROFILES,
    build_youtube_hdr_command, supports_libx265, validate_youtube_source,
    verify_youtube_hdr_file,
)
from youtube_export_analysis import (
    ExportCancelled, apply_youtube_analysis, measure_audio_loudness, measure_content_light,
)
from creative_hdr import active_grade, require_creative_filters, probe_grade_geometry
from creative_hdr_pipeline import grade_command
from creative_hdr_ui import CreativeHDRPanel
from workspace_ui import build_workspace, set_exporting
from gpu_colour import choose_gpu_plan
from gpu_decode import choose_nvidia_decoder


class ConversionThread(QtCore.QThread):
    update_status = QtCore.pyqtSignal(str)
    log_output = QtCore.pyqtSignal(str)

    def __init__(self, ffmpeg_commands, parent=None, verify_output=None, prepare_job=None):
        super().__init__(parent)
        self.ffmpeg_commands = ffmpeg_commands
        self.verify_output = verify_output
        self.prepare_job = prepare_job
        self._process = None
        self._cancel_requested = False
        self.resources = []

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
                    "Status: Converting {}/{}: {}".format(number, total, name))
                self.log_output.emit("Starting {}/{}: {}".format(number, total, name))
                if self.prepare_job is not None:
                    try:
                        ffmpeg_cmd = self.prepare_job(ffmpeg_cmd, self)
                    except ExportCancelled:
                        self._cancel_requested = True
                        break
                    except Exception as exc:
                        if self._cancel_requested:
                            break
                        failed += 1
                        self.log_output.emit(
                            "HDR export analysis failed for {}: {}".format(name, exc))
                        continue
                    if self._cancel_requested:
                        break
                return_code = None
                try:
                    self._process = subprocess.Popen(
                        ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        universal_newlines=True, bufsize=1,
                    )
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
                        self.log_output.emit("Verified YouTube HDR output: {}".format(name))
                    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                        verification_failed = True
                        self.log_output.emit(
                            "YouTube HDR verification failed for {}: {}".format(name, exc))
                if return_code == 0 and not verification_failed:
                    succeeded += 1
                else:
                    failed += 1
                    if verification_failed:
                        self.log_output.emit(
                            "{} was encoded but is NOT verified YouTube HDR-ready.".format(name))
                    else:
                        self.log_output.emit(
                            "Conversion failed for {} (FFmpeg exit code {}).".format(
                                name, return_code if return_code is not None else "unavailable"))
            if self._cancel_requested:
                self.update_status.emit(
                    "Status: Canceled ({} succeeded, {} failed).".format(succeeded, failed))
            elif failed:
                self.update_status.emit(
                    "Status: Finished with errors ({} succeeded, {} failed).".format(succeeded, failed))
            else:
                self.update_status.emit(
                    "Status: Conversion complete ({} succeeded).".format(succeeded))
        except Exception as exc:
            self.update_status.emit("Status: Unexpected conversion error - {}".format(exc))
        finally:
            for resource in self.resources:
                try:
                    resource.close()
                except OSError:
                    pass
            self.resources.clear()

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
        self.setAcceptDrops(True)
        self.default_style = self.styleSheet()
        self.conversion_thread = None
        self.metadata_thread = None
        self.export_active = False
        self._close_after_export = False
        self.init_ui()

    def update_hdr_controls(self, depth):
        is_hdr = depth == "10"
        self.hdr_reference_nits.setEnabled(is_hdr)
        self.embed_metadata.setEnabled(is_hdr)
        self.generate_metadata.setEnabled(is_hdr)
        self.override_metadata.setEnabled(is_hdr)
        self.color_processing_label.setText(
            "10-bit HDR: BT.709 SDR to BT.2020 PQ (automatic)"
            if is_hdr else "8-bit output: SDR only; no HDR metadata")
        self.toggle_override_fields()

    def toggle_override_fields(self):
        creative = self.creative_panel.isChecked()
        self.override_metadata.setEnabled(self.bit_depth.currentText() == "10" and not creative)
        enabled = self.bit_depth.currentText() == "10" and self.override_metadata.isChecked() and not creative
        self.max_cll_input.setEnabled(enabled)
        self.max_fall_input.setEnabled(enabled)

    def init_ui(self):
        build_workspace(self)

    def select_input_file(self):
        if self.batch_mode.isChecked():
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Input Folder")
        else:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select SDR Video", "", "Video Files (*.mp4 *.mov *.mkv)")
        if path:
            self.input_path.setText(path)

    def select_output_file(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self.output_path.setText(path)

    def build_ffmpeg_command(self, ffmpeg_exe, input_path, output_path, use_gpu, source_profile=None):
        bit_depth = self.bit_depth.currentText()
        quality = str(self.cq_slider.value())
        source_profile = source_profile or DEFAULT_SDR_PROFILE
        grade = self.creative_panel.settings()
        if active_grade(grade) and bit_depth != "10":
            raise ValueError("Creative HDR requires 10-bit output; disable it for SDR export.")
        if use_gpu:
            encoder = "hevc_nvenc" if bit_depth == "10" else "h264_nvenc"
            encoder_settings = ["-rc", "constqp", "-qp", quality]
        else:
            encoder = "libx265" if bit_depth == "10" else "libx264"
            encoder_settings = ["-crf", quality]
        pixel_format = ("p010le" if use_gpu else "yuv420p10le") if bit_depth == "10" else "yuv420p"
        ffmpeg_cmd = [ffmpeg_exe, "-y", "-i", input_path]
        if bit_depth == "10":
            ffmpeg_cmd += [
                "-vf", build_sdr_to_pq_filter(source_profile, self.hdr_reference_nits.value(), pixel_format),
                "-color_primaries", "bt2020", "-color_trc", "smpte2084",
                "-colorspace", "bt2020nc", "-color_range", "tv",
            ]
        ffmpeg_cmd += ["-c:v", encoder, "-pix_fmt", pixel_format] + encoder_settings
        if (bit_depth == "10" and encoder == "libx265"
            and source_profile["primaries"] == "bt709"
            and (self.embed_metadata.isChecked() or self.generate_metadata.isChecked())):
            max_cll = self.hdr_reference_nits.value()
            max_fall = 0
            if self.override_metadata.isChecked() and not active_grade(grade):
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
            ffmpeg_cmd += ["-x265-params", x265_sdr_mastering_params(
                self.hdr_reference_nits.value(), max_cll, max_fall)]
        if self.audio_passthrough.isChecked():
            ffmpeg_cmd += ["-c:a", "copy"]
        else:
            ffmpeg_cmd += ["-c:a", "aac", "-b:a", "128k"]
        ffmpeg_cmd.append(output_path)
        return grade_command(ffmpeg_cmd, source_profile, self.hdr_reference_nits.value(), grade)

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
        grade = self.creative_panel.settings()
        creative = active_grade(grade)
        reference_white = self.hdr_reference_nits.value()
        suffix = YOUTUBE_SUFFIX if youtube_mode else ("_HDR" if bit_depth == "10" else "_SDR")
        try:
            if creative:
                grade.validate(reference_white)
                if bit_depth != "10":
                    raise ValueError("Creative HDR requires 10-bit output; disable it for SDR export.")
            jobs = plan_conversion_jobs(input_path, output_path, self.batch_mode.isChecked(), output_suffix=suffix)
        except (ValueError, OSError) as exc:
            self.status_label.setText("Status: {}".format(exc))
            return
        ffmpeg_exe = find_ffmpeg()
        if not ffmpeg_exe:
            self.status_label.setText("Status: FFmpeg not found. Install FFmpeg and add it to PATH.")
            return
        if bit_depth == "10" and not has_zscale(ffmpeg_exe):
            self.status_label.setText(
                "Status: FFmpeg lacks the zscale/zimg colour filter. Install a full FFmpeg build before exporting HDR.")
            return
        ffprobe_exe = find_ffprobe(ffmpeg_exe)
        if creative:
            try:
                if not ffprobe_exe:
                    raise ValueError("Creative HDR requires FFprobe for post-grade light analysis.")
                require_creative_filters(ffmpeg_exe)
            except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
                self.status_label.setText("Status: {}".format(exc))
                return
        if youtube_mode:
            if not ffprobe_exe:
                self.status_label.setText("Status: YouTube HDR requires FFprobe for input/output verification.")
                return
            if not supports_libx265(ffmpeg_exe):
                self.status_label.setText("Status: YouTube HDR requires an FFmpeg build with libx265.")
                return
        elif not ffprobe_exe:
            self.log_output.appendPlainText("Warning: FFprobe not found. Assuming SDR BT.709 limited-range source.")
        use_gpu = False if youtube_mode else self.gpu_checkbox.isChecked()
        if use_gpu:
            try:
                result = subprocess.run([ffmpeg_exe, "-hide_banner", "-encoders"],
                                        capture_output=True, text=True, timeout=15)
                available_encoders = result.stdout if result.returncode == 0 else ""
            except (OSError, subprocess.TimeoutExpired):
                available_encoders = ""
            required_encoder = "hevc_nvenc" if bit_depth == "10" else "h264_nvenc"
            if required_encoder not in available_encoders:
                self.log_output.appendPlainText(
                    "GPU encoder {} unavailable; falling back to CPU.".format(required_encoder))
                use_gpu = False
        if bit_depth == "10" and use_gpu and (creative or self.embed_metadata.isChecked() or self.generate_metadata.isChecked()):
            self.log_output.appendPlainText(
                "Notice: NVENC produces PQ/BT.2020 HDR, but this path does not embed x265 mastering-display SEI. Select CPU/x265 for ST 2086.")
        commands = []
        youtube_preflights = {}
        quality_crf, quality_preset = YOUTUBE_QUALITY_PROFILES[self.youtube_quality.currentText()]
        analyse_light = creative or (youtube_mode and self.youtube_analyse_light.isChecked())
        normalize_audio = youtube_mode and self.youtube_normalize_audio.isChecked()
        accelerate = self.processing_gpu.isChecked() and bit_depth == "10"
        gpu_decoding = self.hardware_decode.isChecked() and bit_depth == "10"
        # Snapshots only. Workers never read changing GUI controls.
        gpu_state = {"attempted": False, "plan": None}
        if creative:
            self.log_output.appendPlainText(
                "Creative HDR enabled: {}. Post-grade MaxCLL/MaxFALL analysis is mandatory; manual light values are not reused. The peak is a virtual BT.2020 mastering target, not a measured display.".format(grade))
        elif youtube_mode and self.override_metadata.isChecked() and analyse_light:
            analyse_light = False
            self.log_output.appendPlainText(
                "Notice: manual HDR light-level overrides take precedence over automatic MaxCLL/MaxFALL analysis.")
        try:
            for source, destination in jobs:
                profile = probe_sdr_profile(ffprobe_exe, source)
                youtube_preflights[source] = (profile, {})
                if youtube_mode:
                    source_info = validate_youtube_source(ffprobe_exe, source, profile)
                    youtube_preflights[source] = (profile, source_info)
                    white_nits = reference_white
                    max_cll, max_fall = white_nits, 0
                    if self.override_metadata.isChecked() and not creative:
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
                    command = build_youtube_hdr_command(
                        ffmpeg_exe, source, destination, profile, white_nits=white_nits,
                        crf=quality_crf, max_cll=max_cll, max_fall=max_fall, preset=quality_preset)
                    commands.append(grade_command(command, profile, white_nits, grade))
                else:
                    if creative:
                        youtube_preflights[source] = (profile, probe_grade_geometry(ffprobe_exe, source))
                    if bit_depth == "10" and profile["primaries"] != "bt709" and not creative:
                        self.log_output.appendPlainText(
                            "Notice: {} has non-BT.709 primaries; static mastering-display metadata will be omitted.".format(os.path.basename(source)))
                    commands.append(self.build_ffmpeg_command(ffmpeg_exe, source, destination, use_gpu, profile))
        except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
            self.status_label.setText("Status: {}".format(exc))
            return
        verify_output = None
        if youtube_mode:
            self.log_output.appendPlainText(
                "YouTube HDR: CPU x265 Main 10 ({} / CRF {} / {}), BT.2020/PQ, AAC stereo 48 kHz. Analyze MaxCLL/FALL: {}; normalize audio: {}. Normal GPU/CRF/audio-passthrough controls are not used.".format(
                    self.youtube_quality.currentText(), quality_crf, quality_preset, analyse_light, normalize_audio))
            verify_output = lambda output: verify_youtube_hdr_file(ffprobe_exe, output)
        prepare_job = None
        if analyse_light or normalize_audio or gpu_decoding:
            def prepare_job(ffmpeg_cmd, worker):
                source = ffmpeg_cmd[ffmpeg_cmd.index("-i") + 1]
                profile, info = youtube_preflights[source]

                def register_process(process):
                    worker._process = process
                    if process is not None and worker._cancel_requested:
                        process.terminate()

                def cancelled():
                    return worker._cancel_requested

                decoder = None
                if gpu_decoding:
                    worker.update_status.emit("Status: Checking GPU decoding...")
                    decoder = choose_nvidia_decoder(
                        ffmpeg_exe, ffprobe_exe, source, profile,
                        log=worker.log_output.emit, cancelled=cancelled,
                        register_process=register_process)
                if accelerate and analyse_light and not gpu_state["attempted"]:
                    gpu_state["attempted"] = True
                    worker.update_status.emit("Status: Checking GPU colour processing...")
                    try:
                        gpu_state["plan"] = choose_gpu_plan(
                            ffmpeg_exe, grade if creative else None, reference_white,
                            log=worker.log_output.emit, cancelled=cancelled,
                            register_process=register_process)
                    except (ValueError, OSError) as exc:
                        worker.log_output.emit("GPU check failed; retaining CPU processing: " + str(exc))
                    if gpu_state["plan"] is not None:
                        worker.resources.append(gpu_state["plan"])
                gpu = gpu_state["plan"]
                light_levels = None
                audio_stats = None
                if analyse_light:
                    worker.update_status.emit("Status: Measuring {}MaxCLL/MaxFALL: {}".format(
                        "post-grade " if creative else "", os.path.basename(source)))
                    try:
                        light_levels = measure_content_light(
                            ffmpeg_exe, source, profile, int(info["width"]), int(info["height"]),
                            reference_white, register_process=register_process,
                            cancelled=cancelled, progress=worker.log_output.emit,
                            grade=grade if creative else None, gpu_plan=gpu, decode_plan=decoder)
                    except (ValueError, OSError) as exc:
                        if cancelled():
                            raise ExportCancelled()
                        if gpu is None and decoder is None:
                            raise
                        worker.log_output.emit(
                            "Accelerated analysis failed; restarting the complete scan on CPU: " + str(exc))
                        # Never combine metadata from a failed GPU scan with a CPU encode.
                        gpu_state["plan"] = gpu = decoder = None
                        light_levels = measure_content_light(
                            ffmpeg_exe, source, profile, int(info["width"]), int(info["height"]),
                            reference_white, register_process=register_process,
                            cancelled=cancelled, progress=worker.log_output.emit,
                            grade=grade if creative else None)
                    worker.log_output.emit(
                        "Measured HDR light levels: MaxCLL={} nits, MaxFALL={} nits ({} frames).".format(*light_levels))
                if normalize_audio and info.get("has_audio", False):
                    worker.update_status.emit("Status: Measuring audio loudness: {}".format(os.path.basename(source)))
                    audio_stats = measure_audio_loudness(
                        ffmpeg_exe, source, register_process=register_process, cancelled=cancelled)
                    if audio_stats is None:
                        worker.log_output.emit("Audio is silent; normalization was skipped.")
                    else:
                        worker.log_output.emit(
                            "Audio input loudness: {} LUFS, true peak {} dBTP. Applying two-pass -16 LUFS / -1.5 dBTP normalization.".format(
                                audio_stats["input_i"], audio_stats["input_tp"]))
                elif normalize_audio:
                    worker.log_output.emit("Source has no audio; normalization was skipped.")
                prepared = apply_youtube_analysis(
                    ffmpeg_cmd, reference_white, light_levels=light_levels,
                    audio_measurements=audio_stats, grade=grade if creative else None)
                if gpu is not None:
                    prepared = gpu.export_command(prepared, profile)
                if decoder is not None:
                    prepared = decoder.apply(prepared)
                worker.log_output.emit(
                    "Processing backend: grade={}; decode={}; video encoder={} (quality/audio unchanged).".format(
                        "OpenCL" if gpu is not None and creative else "CPU",
                        "NVDEC" if decoder is not None else "CPU",
                        prepared[prepared.index("-c:v") + 1]))
                return prepared
        self.conversion_thread = ConversionThread(
            commands, self, verify_output=verify_output, prepare_job=prepare_job)
        self.conversion_thread.update_status.connect(self.status_label.setText)
        self.conversion_thread.log_output.connect(self.log_output.appendPlainText)
        self.conversion_thread.finished.connect(self.on_conversion_finished)
        self.start_btn.setEnabled(False)
        self.youtube_btn.setEnabled(False)
        set_exporting(self, True)
        self.conversion_thread.start()

    def cancel_conversion(self):
        reply = QtWidgets.QMessageBox.question(
            self, 'Cancel Conversion', 'Are you sure you want to cancel the conversion?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            if self.conversion_thread and self.conversion_thread.isRunning():
                self.conversion_thread.cancel()
                self.status_label.setText("Status: Cancel requested...")

    def on_conversion_finished(self):
        thread = self.conversion_thread
        self.conversion_thread = None
        if thread is not None:
            thread.deleteLater()
        self.start_btn.setEnabled(True)
        self.youtube_btn.setEnabled(True)
        set_exporting(self, False)
        if self._close_after_export:
            self.close()

    def dragEnterEvent(self, event):
        if not self.export_active and event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) == 1 and urls[0].isLocalFile():
                event.acceptProposedAction()

    def dropEvent(self, event):
        if self.export_active:
            return
        urls = event.mimeData().urls()
        if len(urls) == 1 and urls[0].isLocalFile():
            path = urls[0].toLocalFile()
            self.batch_mode.setChecked(os.path.isdir(path))
            self.input_path.setText(path)
            event.acceptProposedAction()

    def closeEvent(self, event):
        if self.conversion_thread and self.conversion_thread.isRunning():
            if not self._close_after_export:
                reply = QtWidgets.QMessageBox.question(
                    self, 'Export is running', 'Cancel the export and close when it has stopped?',
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
                if reply == QtWidgets.QMessageBox.Yes:
                    self._close_after_export = True
                    self.conversion_thread.cancel()
                    self.preview.shutdown()
                    self.status_label.setText('Stopping export before closing...')
            event.ignore()
            return
        self.source_timer.stop()
        self.preview.shutdown()
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    converter = SDRtoHDRConverter()
    converter.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
