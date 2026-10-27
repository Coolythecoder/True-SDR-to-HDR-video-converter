"""Compile the exact scalar CUDA maths for host tests against real FFmpeg.

Host execution is test-only, not a production fallback and not NVIDIA testing.
"""
import ctypes
from dataclasses import replace
from pathlib import Path
import shutil
import sys
import subprocess
import tempfile
import unittest
import numpy as np
from creative_hdr import CreativeHDRSettings, build_creative_filter
from cuda_colour import colour_source, cuda_source
from hdr_processing import DEFAULT_SDR_PROFILE, build_sdr_to_linear_filter


@unittest.skipUnless(sys.platform != 'win32' and shutil.which('g++') and shutil.which('ffmpeg'), 'Needs host C++ compiler and FFmpeg')
class ScalarMathTests(unittest.TestCase):
    def compile(self, grade=CreativeHDRSettings(), white=200):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        source = Path(td.name)/'math.cpp'
        binary = Path(td.name)/'math.so'
        source.write_text(colour_source(grade, white) + r'''
extern "C" void apply(float* p,int n,int mode,float white) {
 for(int i=0;i<n;i++) {V v=vec(p[3*i],p[3*i+1],p[3*i+2]);
 if(mode==1) v=linear709(v.x,v.y,v.z);
 if(mode!=2) v=grade_pixel(v); else v=vec(pq(v.x,white),pq(v.y,white),pq(v.z,white));
 p[3*i]=v.x;p[3*i+1]=v.y;p[3*i+2]=v.z;
 }}
''', encoding='utf-8')
        build = subprocess.run(['g++','-shared','-fPIC','-O2','-ffp-contract=off',str(source),'-o',str(binary)],capture_output=True)
        self.assertEqual(build.returncode,0,build.stderr.decode())
        lib = ctypes.CDLL(str(binary))
        fn = lib.apply
        fn.argtypes = [ctypes.POINTER(ctypes.c_float),ctypes.c_int,ctypes.c_int,ctypes.c_float]
        def apply(p,mode=0):
            out = np.array(p,dtype=np.float32,copy=True,order='C')
            fn(out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),out.size//3,mode,white)
            return out
        return apply

    def run_ffmpeg(self, raw, fmt, w, h, filt):
        p = subprocess.run(['ffmpeg','-v','error','-f','rawvideo','-pixel_format',fmt,
                            '-video_size',f'{w}x{h}','-i','pipe:0','-vf',filt,'-frames:v','1',
                            '-pix_fmt','gbrpf32le','-f','rawvideo','pipe:1'],input=raw,capture_output=True)
        self.assertEqual(p.returncode,0,p.stderr.decode())
        return np.frombuffer(p.stdout,dtype='<f4').reshape(3,h,w)[[2,0,1]].transpose(1,2,0)

    def test_all_grade_controls_match_original_cpu_equations(self):
        rng=np.random.default_rng(83)
        p=rng.uniform(-.03,2,(18,34,3)).astype('f4')
        p[0]=0
        for grade in (CreativeHDRSettings(),CreativeHDRSettings(enabled=True),
            CreativeHDRSettings(enabled=True,peak_nits=4000,strength=.9,knee=.65,rolloff=.7,
              exposure=.7,contrast=1.2,midtones=.3,shadows=-.5,black_floor=.2,saturation=1.4,
              vibrance=.7,temperature=.6,tint=-.4),
            CreativeHDRSettings(enabled=True,strength=0,saturation=0)):
            with self.subTest(grade=grade):
                actual=self.compile(grade)(p)
                ref=self.run_ffmpeg(p.transpose(2,0,1)[[1,2,0]].tobytes(),'gbrpf32le',34,18,
                                    build_creative_filter(grade).lstrip(',') or 'null')
                np.testing.assert_allclose(actual,ref,atol=3e-5,rtol=3e-4)

    def test_709_linear_transform_matches_zscale_for_neutrals_and_colours(self):
        apply=self.compile()
        for full in (False,True):
            for depth in (8,10):
                mult=1 if depth==8 else 4
                maximum=(1<<depth)-1
                for yy,cb,cr in ((16,128,128),(126,128,128),(235,128,128),(96,96,176),(160,180,80)):
                    y,u,v=yy*mult,cb*mult,cr*mult
                    dtype='u1' if depth==8 else '<u2'
                    raw=(np.full(128,y,dtype=dtype).tobytes()+np.full(32,u,dtype=dtype).tobytes()+np.full(32,v,dtype=dtype).tobytes())
                    profile=dict(DEFAULT_SDR_PROFILE,range='full' if full else 'limited')
                    ref=self.run_ffmpeg(raw,'yuv420p' if depth==8 else 'yuv420p10le',16,8,
                                        build_sdr_to_linear_filter(profile))[0,0]
                    val=[y/maximum if full else (y-16*mult)/(219*mult),
                         (u-128*mult)/(maximum if full else 224*mult),
                         (v-128*mult)/(maximum if full else 224*mult)]
                    actual=apply(np.array(val).reshape(1,3),mode=1)[0]
                    np.testing.assert_allclose(actual,ref,atol=3e-5,rtol=3e-4)

    def test_pq_known_nits_match_zscale(self):
        for white in (100,200,400):
            apply=self.compile(white=white)
            for nits in (0,1,100,200,600,1000,4000):
                p=np.full((8,16,3),nits/white,dtype='f4')
                ref=self.run_ffmpeg(p.transpose(2,0,1).tobytes(),'gbrpf32le',16,8,
                    f'zscale=pin=bt2020:tin=linear:min=gbr:rin=full:p=bt2020:t=smpte2084:m=gbr:r=full:npl={white}')
                np.testing.assert_allclose(apply(p,mode=2),ref,atol=5e-5,rtol=3e-4)


class SourceGenerationTests(unittest.TestCase):
    def test_native_source_uses_same_generated_grade_and_no_fast_math(self):
        code=cuda_source(CreativeHDRSettings(enabled=True))
        self.assertIn('grade_pixel(linear709',code)
        self.assertIn('convert8',code)
        self.assertIn('convert16',code)
        self.assertIn('pack_p010',code)
        self.assertNotIn('read_image',code)
        self.assertIn('powf',code)


if __name__=='__main__': unittest.main()
