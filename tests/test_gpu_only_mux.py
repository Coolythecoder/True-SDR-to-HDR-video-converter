"""Exercise actual FFmpeg MP4 mux/timing/SEI validation, using a CPU fixture.

The fixture encoder is not a replacement for a physical NVENC test.
"""
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from gpu_only_export import GPUOnlySettings, SourceInfo, mux_command, verify_output, _run


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'Needs FFmpeg/FFprobe')
class RealMuxTests(unittest.TestCase):
    def command(self,args):
        p=subprocess.run(args,capture_output=True,timeout=60)
        self.assertEqual(p.returncode,0,p.stderr.decode(errors='replace')[-3000:])
        return p.stdout

    def test_hdr_metadata_and_exact_cfr_survive_video_copy_mux(self):
        for fps in (Fraction(144),Fraction(24000,1001)):
            with self.subTest(fps=fps),tempfile.TemporaryDirectory() as td:
                raw=Path(td)/'native.hevc';audio=Path(td)/'audio.m4a';out=Path(td)/'mux.mp4'
                self.command(['ffmpeg','-v','error','-f','lavfi','-i',
                  'color=white:size=128x64:rate='+str(fps),'-frames:v','6','-pix_fmt','yuv420p10le',
                  '-c:v','libx265','-preset','ultrafast','-x265-params',
                  'pools=1:frame-threads=1:bframes=0:repeat-headers=1:hdr10=1:master-display=G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)L(6000000,0):max-cll=400,100',
                  '-f','hevc',str(raw)])
                self.command(['ffmpeg','-v','error','-f','lavfi','-i','anullsrc=r=48000:cl=stereo',
                              '-t','0.1','-c:a','aac',str(audio)])
                info=SourceInfo(128,64,10,fps,Fraction(1,14400),False,False)
                settings=GPUOnlySettings(copy_audio=True)
                self.command(mux_command('ffmpeg',raw,audio,out,info,settings))
                verify_output('ffprobe',out,info,6,400,100,lambda:False,lambda p:None)
                pts=json.loads(self.command(['ffprobe','-v','error','-select_streams','v:0',
                       '-show_packets','-show_streams','-of','json',str(out)]))
                tb=Fraction(pts['streams'][0]['time_base'])
                times=[int(p['pts'])*tb for p in pts['packets']]
                self.assertEqual(times,[i/fps for i in range(6)])
                # Verification must not accept substituted light metadata.
                with self.assertRaises(ValueError):
                    verify_output('ffprobe',out,info,6,1000,100,lambda:False,lambda p:None)


if __name__=='__main__':unittest.main()
