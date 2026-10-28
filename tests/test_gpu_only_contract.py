import copy
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np
from creative_hdr import CreativeHDRSettings
from gpu_only_export import (GPUOnlySettings, SourceInfo, validate_source, validate_timestamps,
    sei_messages, encoder_parameters, mux_command, InputProbeFrame, EncoderFrame, _run, export_gpu_only)
from youtube_export_analysis import ExportCancelled


def probe():
    return {'streams':[dict(codec_type='video', codec_name='hevc',color_space='bt709',
      color_primaries='bt709',color_transfer='bt709',pix_fmt='yuv420p10le',width=3840,height=2160,
      field_order='progressive',sample_aspect_ratio='1:1',avg_frame_rate='144/1',time_base='1/14400',
      color_range='tv',chroma_location='left')]}


class ContractTests(unittest.TestCase):
    def test_supported_4k144_is_not_retimed(self):
        info=validate_source(probe())
        self.assertEqual(info.fps,Fraction(144))
        self.assertEqual(validate_timestamps([200,0,100,300],info),4)

    def test_fractional_cfr_rounding_accepted(self):
        info=replace(validate_source(probe()),fps=Fraction(24000,1001),time_base=Fraction(1,1000))
        self.assertEqual(validate_timestamps([round(i*1001/24) for i in range(200)],info),200)

    def test_vfr_gaps_duplicates_offsets_are_rejected(self):
        info=validate_source(probe())
        for pts in ([],[0,100,300],[0,100,100],[100,200,300],[0,50,200]):
            with self.subTest(pts=pts),self.assertRaises(ValueError): validate_timestamps(pts,info)

    def test_input_contract_rejects_unhandled_colour_geometry_and_timing(self):
        for key,value in (('color_transfer','smpte2084'),('color_primaries','bt2020'),
          ('color_range','unknown'),('pix_fmt','yuv444p'),('width',3839),('field_order','tt'),
          ('sample_aspect_ratio','4:3'),('avg_frame_rate','0/1'),('codec_name','vp9'),
          ('chroma_location','topleft')):
            data=probe();data['streams'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_source(data)
        data=probe();data['streams'][0]['tags']={'rotate':'90'}
        with self.assertRaises(ValueError):validate_source(data)

    def test_settings_are_frozen_and_invalid_values_fail(self):
        for kwargs in ({'qp':52},{'qp':True},{'device':-1},{'white_nits':1000},{'copy_audio':'yes'}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):GPUOnlySettings(**kwargs).validate()

    def test_sei_payloads_are_big_endian_and_use_measured_light(self):
        import struct
        s=GPUOnlySettings(grade=CreativeHDRSettings(enabled=True,peak_nits=1000))
        msgs=sei_messages(s,749,123)
        self.assertEqual(msgs[0][0],{'sei_type':137})
        self.assertEqual(len(msgs[0][1]),24)
        self.assertEqual(struct.unpack('>8H2I',bytes(msgs[0][1]))[-2:],(10000000,0))
        self.assertEqual(struct.unpack('>2H',bytes(msgs[1][1])),(749,123))
        with self.assertRaises(ValueError):sei_messages(s,100,101)

    def test_encoder_and_mux_never_invoke_cpu_video_filters_or_x265(self):
        info=validate_source(probe());settings=GPUOnlySettings(qp=12,copy_audio=True)
        params=encoder_parameters(info,settings)
        self.assertEqual(params['constqp'],12)
        self.assertEqual(params['fps'],144)
        cmd=mux_command('ffmpeg','raw.hevc','source.mp4','out.mp4',info,settings)
        self.assertEqual(cmd[cmd.index('-c:v')+1],'copy')
        self.assertEqual(cmd[cmd.index('-c:a')+1],'copy')
        self.assertNotIn('-vf',cmd);self.assertNotIn('libx265',cmd)
        self.assertEqual(cmd[cmd.index('-r')+1],'144')
        self.assertIn('chroma_sample_loc_type=1',cmd[cmd.index('-bsf:v')+1])

    def test_p010_encoder_plane_shapes_and_byte_strides(self):
        height, width = 18, 34
        raw = np.zeros((height * 3 // 2, width), dtype=np.uint16)
        raw[0, 0], raw[height, 0], raw[height, 1] = 0xA1B2, 0xC3D4, 0xE5F6

        # CUDA grading still consumes native uint16 planes; only NVENC sees
        # the byte-typed CAI interface (the <u2 interface was rejected).
        probe_y, probe_uv = InputProbeFrame(raw, height, width).cuda()
        self.assertEqual(probe_y.dtype, np.dtype('uint16'))
        self.assertEqual(probe_uv.strides, (width * 2, 4, 2))

        encoder = EncoderFrame(raw, height, width)
        y, uv = encoder.cuda()
        self.assertEqual(y.shape, (height, width, 1))
        self.assertEqual(uv.shape, (height // 2, width // 2, 2))
        self.assertEqual(y.dtype.str, '|u1')
        self.assertEqual(uv.dtype.str, '|u1')
        self.assertEqual(y.strides, (width * 2, 2, 1))
        self.assertEqual(uv.strides, (width * 2, 4, 2))
        self.assertEqual(y.ctypes.data, raw.ctypes.data)
        self.assertEqual(uv.ctypes.data - raw.ctypes.data, height * width * 2)
        self.assertTrue(np.shares_memory(raw, y))
        self.assertTrue(np.shares_memory(raw, uv))
        self.assertEqual(int(y[0,0,0]), 0xB2)
        self.assertEqual(tuple(int(v) for v in uv[0,0]), (0xD4, 0xF6))
        # No re-encoding, endian swap or copy of the 16-bit P010 words.
        self.assertEqual(encoder.byte_rows.tobytes(), raw.tobytes())

    def test_p010_encoder_does_not_silently_accept_wrong_layout(self):
        for raw in (np.zeros((27,34), dtype=np.uint8),
                    np.zeros((27,34), dtype=np.uint16)[:, ::-1],
                    np.zeros((27,33), dtype=np.uint16)):
            with self.subTest(shape=raw.shape, strides=raw.strides, dtype=raw.dtype), self.assertRaises(ValueError):
                EncoderFrame(raw, 18, 34)

    def test_cancel_does_not_spawn_cpu_fallback(self):
        with self.assertRaises(ExportCancelled):
            _run([sys.executable,'-c','import time;time.sleep(10)'],lambda:True,lambda p:None)

    def test_missing_input_or_in_place_export_leaves_files_untouched(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'input.mp4';p.write_bytes(b'unchanged')
            with self.assertRaises(ValueError): export_gpu_only('ffmpeg','ffprobe',p,p,GPUOnlySettings())
            self.assertEqual(p.read_bytes(),b'unchanged')

    def test_source_has_no_uncompressed_frame_download_in_bulk_loop(self):
        import inspect
        text=inspect.getsource(export_gpu_only)
        self.assertNotIn('asnumpy(',text)
        self.assertNotIn('hwdownload',text)
        self.assertNotIn('zscale=',text)
        self.assertIn('use_device_memory=True',text)
        self.assertIn('os.replace(complete, destination)',text)


if __name__=='__main__':unittest.main()
