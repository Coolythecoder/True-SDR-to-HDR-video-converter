"""End-to-end OpenCL media checks. Software OpenCL proves maths, not GPU speed."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np
from creative_hdr import CreativeHDRSettings
from creative_hdr_pipeline import grade_command
from gpu_colour import GPUPlan, kernel_source, _path_option
from gpu_decode import DecodePlan, choose_nvidia_decoder
from grading_preview import render_command
from hdr_processing import DEFAULT_SDR_PROFILE
from youtube_hdr_export import build_youtube_hdr_command, verify_youtube_hdr_file
from youtube_export_analysis import measure_content_light, apply_youtube_analysis, ExportCancelled

FFMPEG = shutil.which('ffmpeg')
FFPROBE = shutil.which('ffprobe')
DEVICE = os.environ.get('HDR_TEST_OPENCL_DEVICE')


class DecodeContractTests(unittest.TestCase):
    def test_decoder_never_substitutes_encoder_or_changes_timing(self):
        original = build_youtube_hdr_command('ffmpeg','input.mp4','out.mp4',
                    DEFAULT_SDR_PROFILE,crf=12,preset='slow')
        result = DecodePlan('nv12').apply(original)
        self.assertIn('libx265', result)
        self.assertNotIn('hevc_nvenc', result)
        index = result.index('-hwaccel')
        del result[index:index+4]
        result[result.index('-vf')+1] = original[original.index('-vf')+1]
        self.assertEqual(result, original)

    def test_unsupported_format_is_not_forced_into_decoder(self):
        data = json.dumps({'streams':[{'codec_name':'hevc','pix_fmt':'yuv444p12le'}]}).encode()
        with mock.patch('gpu_decode._probe_run', return_value=data) as probe:
            self.assertIsNone(choose_nvidia_decoder('ffmpeg','ffprobe','in',DEFAULT_SDR_PROFILE))
            self.assertEqual(probe.call_count, 1)

    def test_device_failure_returns_software_with_reason(self):
        logs = []
        data = json.dumps({'streams':[{'codec_name':'hevc','pix_fmt':'yuv420p10le'}]}).encode()
        with mock.patch('gpu_decode._probe_run', side_effect=[data,ValueError('CUDA unavailable')]):
            self.assertIsNone(choose_nvidia_decoder('f','p','in',DEFAULT_SDR_PROFILE,log=logs.append))
        self.assertIn('software', logs[-1])

    def test_decoder_difference_is_rejected(self):
        data = json.dumps({'streams':[{'codec_name':'h264','pix_fmt':'yuv420p'}]}).encode()
        a = np.zeros(64*64*3,dtype='<f4').tobytes()
        b = np.ones(64*64*3,dtype='<f4').tobytes()
        with mock.patch('gpu_decode._probe_run', side_effect=[data,a,b]):
            self.assertIsNone(choose_nvidia_decoder('f','p','in',DEFAULT_SDR_PROFILE))

    def test_decoder_cancel_propagates(self):
        with mock.patch('gpu_decode._probe_run', side_effect=ExportCancelled()):
            with self.assertRaises(ExportCancelled):
                choose_nvidia_decoder('f','p','in',DEFAULT_SDR_PROFILE)

    def test_mismatched_analysis_snapshot_rejected_before_spawn(self):
        plan = GPUPlan(CreativeHDRSettings(enabled=True))
        try:
            with mock.patch('youtube_export_analysis.subprocess.Popen') as spawn:
                with self.assertRaisesRegex(ValueError,'snapshot'):
                    measure_content_light('f','in',DEFAULT_SDR_PROFILE,32,32,200,gpu_plan=plan)
                spawn.assert_not_called()
        finally:
            plan.close()


@unittest.skipUnless(FFMPEG and FFPROBE and DEVICE, 'Requires explicit OpenCL device and FFmpeg')
class OpenCLMediaTests(unittest.TestCase):
    def test_actual_scan_encode_metadata_and_luminance(self):
        with tempfile.TemporaryDirectory() as directory:
            source = str(Path(directory)/'source.mp4')
            output = str(Path(directory)/'gpu_HDR.mp4')
            made = subprocess.run([FFMPEG,'-v','error','-y','-f','lavfi','-i',
                'color=c=white:s=1280x720:r=60:d=0.05','-c:v','libx264','-preset','ultrafast',
                '-pix_fmt','yuv420p','-color_primaries','bt709','-color_trc','bt709',
                '-colorspace','bt709','-color_range','tv',source],capture_output=True,timeout=30)
            self.assertEqual(made.returncode,0,made.stderr)
            grade = CreativeHDRSettings(enabled=True)
            plan = GPUPlan(grade,200,DEVICE)
            try:
                cpu = measure_content_light(FFMPEG,source,DEFAULT_SDR_PROFILE,1280,720,200,grade=grade)
                gpu = measure_content_light(FFMPEG,source,DEFAULT_SDR_PROFILE,1280,720,200,grade=grade,gpu_plan=plan)
                self.assertEqual(cpu[2],3)
                self.assertEqual(cpu[2],gpu[2])
                np.testing.assert_allclose(gpu[:2],cpu[:2],atol=1,rtol=0)
                self.assertGreater(gpu[0],390)
                cmd = grade_command(build_youtube_hdr_command(FFMPEG,source,output,DEFAULT_SDR_PROFILE,
                    crf=12,preset='slow'),DEFAULT_SDR_PROFILE,200,grade)
                cmd = apply_youtube_analysis(cmd,200,light_levels=gpu,grade=grade)
                cmd = plan.export_command(cmd,DEFAULT_SDR_PROFILE)
                self.assertEqual(cmd[cmd.index('-c:v')+1],'libx265')
                encoded = subprocess.run(cmd,capture_output=True,timeout=120)
                self.assertEqual(encoded.returncode,0,encoded.stderr.decode(errors='replace')[-3000:])
                self.assertTrue(verify_youtube_hdr_file(FFPROBE,output))
                data = subprocess.run([FFPROBE,'-v','error','-show_streams','-show_frames',
                    '-of','json',output],capture_output=True,timeout=30)
                self.assertEqual(data.returncode,0,data.stderr)
                info = json.loads(data.stdout)
                video = info['streams'][0]
                self.assertEqual(video['nb_frames'],'3')
                self.assertEqual(video['avg_frame_rate'],'60/1')
                side = {d['side_data_type']:d for f in info['frames'] for d in f.get('side_data_list',[])}
                self.assertEqual(side['Content light level metadata']['max_content'],gpu[0])
                self.assertEqual(side['Content light level metadata']['max_average'],gpu[1])
                decoded = subprocess.run([FFMPEG,'-v','error','-i',output,'-frames:v','1','-vf',
                    'zscale=rin=limited:min=bt2020nc:pin=bt2020:tin=smpte2084:t=linear:m=gbr:p=bt2020:r=full:npl=200,format=gbrpf32le',
                    '-pix_fmt','gbrpf32le','-f','rawvideo','pipe:1'],capture_output=True,timeout=30)
                self.assertEqual(decoded.returncode,0,decoded.stderr)
                self.assertAlmostEqual(float(np.frombuffer(decoded.stdout,dtype='<f4').mean())*200,400,delta=8)
            finally:
                plan.close()

    def test_preview_viewing_transform_matches_cpu(self):
        grade = CreativeHDRSettings(enabled=True,temperature=.2,saturation=1.1,exposure=.1)
        pixels = np.random.default_rng(123).uniform(0,1.3,(3,18,32)).astype('<f4')
        plan = GPUPlan(grade,200,DEVICE)
        try:
            cpu = subprocess.run(render_command(FFMPEG,32,18,grade),input=pixels.tobytes(),capture_output=True,timeout=30)
            gpu = subprocess.run(render_command(FFMPEG,32,18,grade,gpu_plan=plan),input=pixels.tobytes(),capture_output=True,timeout=60)
            self.assertEqual(cpu.returncode,0,cpu.stderr)
            self.assertEqual(gpu.returncode,0,gpu.stderr)
            self.assertEqual(len(cpu.stdout),32*18*6)
            diff = np.abs(np.frombuffer(cpu.stdout,dtype=np.uint8).astype(np.int16)-np.frombuffer(gpu.stdout,dtype=np.uint8))
            self.assertLessEqual(diff.max(),2)
            self.assertLessEqual(diff.mean(),.15)
        finally:
            plan.close()

    def test_shader_filename_escaping(self):
        plan = GPUPlan(CreativeHDRSettings(enabled=True),200,DEVICE)
        try:
            # Exercise filter parsing of the characters common in Windows paths.
            path = Path(plan._temp.name)/"drive: Joe's [HDR], colour ü.cl"
            path.write_text(kernel_source(plan.grade,200),encoding='utf-8')
            plan.shader = path
            result = subprocess.run(plan.raw_command(FFMPEG,16,16),
                input=np.ones((3,16,16),dtype='<f4').tobytes(),capture_output=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stderr.decode(errors='replace'))
        finally:
            plan.close()


if __name__ == '__main__':
    unittest.main()
