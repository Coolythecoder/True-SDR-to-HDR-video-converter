import os
from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
import unittest
from unittest import mock

import numpy as np
from creative_hdr import CreativeHDRSettings, build_creative_filter
from creative_hdr_pipeline import grade_command
from gpu_colour import GPUPlan, GPU_DEVICES, kernel_source, tile_light_levels, choose_gpu_plan
from hdr_processing import DEFAULT_SDR_PROFILE
from youtube_hdr_export import build_youtube_hdr_command
from youtube_export_analysis import linear_frame_light_levels, ExportCancelled

FFMPEG = shutil.which('ffmpeg')
DEVICE = os.environ.get('HDR_TEST_OPENCL_DEVICE')


class GPURecipeTests(unittest.TestCase):
    def test_youtube_settings_are_unchanged(self):
        grade = CreativeHDRSettings(enabled=True)
        original = grade_command(build_youtube_hdr_command(
            'ffmpeg','input.mp4','out.mp4',DEFAULT_SDR_PROFILE,
            crf=12,preset='slow',max_cll=400,max_fall=100),DEFAULT_SDR_PROFILE,200,grade)
        plan = GPUPlan(grade)
        try:
            result = plan.export_command(original,DEFAULT_SDR_PROFILE)
            # Only the filter and global GPU-device options may change.
            result[1:5] = []
            result[result.index('-vf')+1] = original[original.index('-vf')+1]
            self.assertEqual(result,original)
            self.assertIn('libx265',result)
            self.assertNotIn('hevc_nvenc',result)
        finally:
            plan.close()

    def test_accurate_export_does_not_replace_colour_transform(self):
        cmd=build_youtube_hdr_command('ffmpeg','a','b.mp4',DEFAULT_SDR_PROFILE)
        plan=GPUPlan()
        try:
            self.assertEqual(plan.export_command(cmd,DEFAULT_SDR_PROFILE),cmd)
        finally:
            plan.close()

    def test_no_production_cpu_opencl_device_selected(self):
        self.assertTrue(all('device_type=gpu' in x for x in GPU_DEVICES))

    def test_invalid_grade_or_device_fails(self):
        with self.assertRaises(ValueError):
            kernel_source(CreativeHDRSettings(enabled=True,saturation=float('nan')))
        with self.assertRaises(ValueError):
            GPUPlan(device='bad;selector')

    def test_shader_lifetime_and_float_formats(self):
        plan=GPUPlan(CreativeHDRSettings(enabled=True))
        path=plan.shader
        self.assertTrue(path.is_file())
        recipe=plan.filter()
        self.assertIn('grayf32le',recipe)
        self.assertIn('gbrpf32le',recipe)
        self.assertNotIn('rgb24',recipe)
        plan.close()
        self.assertFalse(path.exists())

    def test_no_gpu_falls_back_without_claiming_acceleration(self):
        logs=[]
        with mock.patch('gpu_colour._probe_run',side_effect=[b'expected']+[ValueError('no GPU')]*3):
            self.assertIsNone(choose_gpu_plan('ffmpeg',CreativeHDRSettings(),200,log=logs.append))
        self.assertIn('CPU',logs[-1])

    def test_cancel_is_not_swallowed_as_device_failure(self):
        with mock.patch('gpu_colour._probe_run',side_effect=ExportCancelled()):
            with self.assertRaises(ExportCancelled):
                choose_gpu_plan('ffmpeg',CreativeHDRSettings(),200)

    def test_reduction_rejects_nan_and_wrong_length(self):
        with self.assertRaises(ValueError):
            tile_light_levels(np.array([1,np.nan],dtype='<f4').tobytes(),16,16,200)
        with self.assertRaises(ValueError):
            tile_light_levels(b'bad',16,16,200)


@unittest.skipUnless(FFMPEG and DEVICE,'Requires an explicitly selected OpenCL test device')
class OpenCLParityTests(unittest.TestCase):
    def compare_grade(self, grade, white=200):
        width,height=34,18  # Includes partial reduction tiles.
        pixels=np.random.default_rng(89).uniform(-.03,2,(3,height,width)).astype('<f4')
        pixels[:,0,:]=np.linspace(0,4,width)
        pixels[:,1,:]=0
        plan=GPUPlan(grade,white,DEVICE)
        try:
            command=[FFMPEG,'-v','error','-f','rawvideo','-pixel_format','gbrpf32le',
                     '-video_size',f'{width}x{height}','-i','pipe:0','-vf',
                     build_creative_filter(grade,white).lstrip(',') or 'null',
                     '-frames:v','1','-pix_fmt','gbrpf32le','-f','rawvideo','pipe:1']
            cpu=subprocess.run(command,input=pixels.tobytes(),capture_output=True,timeout=30)
            self.assertEqual(cpu.returncode,0,cpu.stderr.decode(errors='replace'))
            gpu=subprocess.run(plan.raw_command(FFMPEG,width,height),input=pixels.tobytes(),capture_output=True,timeout=60)
            self.assertEqual(gpu.returncode,0,gpu.stderr.decode(errors='replace'))
            a=np.frombuffer(cpu.stdout,dtype='<f4').reshape(pixels.shape)
            b=np.frombuffer(gpu.stdout,dtype='<f4').reshape(pixels.shape)
            np.testing.assert_allclose(b,a,rtol=3e-4,atol=3e-5)
            reduced=subprocess.run(plan.raw_command(FFMPEG,width,height,stats=True),input=pixels.tobytes(),capture_output=True,timeout=60)
            self.assertEqual(reduced.returncode,0,reduced.stderr.decode(errors='replace'))
            peak,mean=tile_light_levels(reduced.stdout,width,height,white)
            expected=linear_frame_light_levels(a,white)
            np.testing.assert_allclose((peak,mean),expected,rtol=3e-4,atol=.01)
        finally:
            plan.close()

    def test_identity_retains_float_headroom(self):
        self.compare_grade(CreativeHDRSettings())

    def test_creative_defaults(self):
        self.compare_grade(CreativeHDRSettings(enabled=True))

    def test_all_controls_and_4000_nit_headroom(self):
        self.compare_grade(CreativeHDRSettings(enabled=True,peak_nits=4000,
            strength=.9,knee=.65,rolloff=.7,exposure=.7,contrast=1.2,
            midtones=.3,shadows=-.5,black_floor=.2,saturation=1.4,vibrance=.7,
            temperature=.6,tint=-.4),white=150)

    def test_monochrome_and_zero_strength(self):
        self.compare_grade(CreativeHDRSettings(enabled=True,strength=0,saturation=0))

    def test_runtime_capability_gate_accepts_valid_backend(self):
        with mock.patch('gpu_colour.GPU_DEVICES',(DEVICE,)):
            plan=choose_gpu_plan(FFMPEG,CreativeHDRSettings(enabled=True),200)
        self.assertIsNotNone(plan)
        if plan:
            plan.close()


if __name__=='__main__':
    unittest.main()
