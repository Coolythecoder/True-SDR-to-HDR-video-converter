import tempfile
import unittest
from pathlib import Path
from unittest import mock

from conversion_paths import find_ffmpeg, plan_conversion_jobs


class ConversionPathsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.src = self.root / 'source'
        self.dst = self.root / 'output'
        self.src.mkdir()
        self.dst.mkdir()

    def video(self, name):
        path = self.src / name
        path.write_bytes(b'placeholder')
        return path

    def test_single_conversion_generates_hdr_filename(self):
        source = self.video('movie.mov')
        self.assertEqual(
            plan_conversion_jobs(str(source), str(self.dst), False),
            [(str(source), str(self.dst / 'movie_HDR.mp4'))],
        )

    def test_previous_explicit_mp4_file_path_still_works(self):
        source = self.video('movie.mkv')
        target = self.dst / 'chosen.mp4'
        self.assertEqual(plan_conversion_jobs(str(source), str(target), False),
                         [(str(source), str(target))])

    def test_batch_handles_video_files_not_the_directory(self):
        mov = self.video('a.MOV')
        mkv = self.video('b.mkv')
        self.video('not_a_video.txt')
        (self.src / 'nested.mp4').mkdir()
        jobs = plan_conversion_jobs(str(self.src), str(self.dst), True)
        self.assertEqual([x[0] for x in jobs], [str(mov), str(mkv)])
        self.assertEqual([x[1] for x in jobs],
                         [str(self.dst / 'a_HDR.mp4'), str(self.dst / 'b_HDR.mp4')])

    def test_batch_same_folder_skips_previous_hdr_outputs(self):
        source = self.video('movie.mp4')
        self.video('movie_HDR.mp4')
        jobs = plan_conversion_jobs(str(self.src), str(self.src), True)
        self.assertEqual(jobs, [(str(source), str(self.src / 'movie_HDR.mp4'))])

    def test_duplicate_stems_have_distinct_output_names(self):
        self.video('clip.mov')
        self.video('clip.mkv')
        jobs = plan_conversion_jobs(str(self.src), str(self.dst), True)
        outputs = [output for _, output in jobs]
        self.assertEqual(len(set(outputs)), 2)
        self.assertTrue(all(out.endswith('.mp4') for out in outputs))

    def test_empty_batch_directory_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'No supported videos'):
            plan_conversion_jobs(str(self.src), str(self.dst), True)

    def test_batch_requires_folder(self):
        source = self.video('movie.mp4')
        with self.assertRaisesRegex(ValueError, 'input folder'):
            plan_conversion_jobs(str(source), str(self.dst), True)

    def test_missing_ffmpeg_path_is_not_treated_as_video(self):
        with self.assertRaisesRegex(ValueError, 'existing MP4'):
            plan_conversion_jobs(str(self.src / 'missing.mp4'), str(self.dst), False)

    def test_missing_output_directory_is_rejected(self):
        source = self.video('movie.mp4')
        with self.assertRaisesRegex(ValueError, 'output folder'):
            plan_conversion_jobs(str(source), str(self.root / 'missing'), False)

    def test_refuses_to_overwrite_its_input(self):
        source = self.video('movie.mp4')
        with self.assertRaisesRegex(ValueError, 'cannot overwrite'):
            plan_conversion_jobs(str(source), str(source), False)

    @mock.patch('conversion_paths.shutil.which', return_value='/usr/bin/ffmpeg')
    def test_ffmpeg_on_path(self, _which):
        self.assertEqual(find_ffmpeg(), '/usr/bin/ffmpeg')

    @mock.patch('conversion_paths.sys.platform', 'darwin')
    @mock.patch('conversion_paths.shutil.which', return_value=None)
    @mock.patch('conversion_paths.os.path.isfile', side_effect=lambda p: p == '/opt/homebrew/bin/ffmpeg')
    @mock.patch('conversion_paths.os.access', return_value=True)
    def test_homebrew_fallback_for_macos_gui(self, _access, _file, _which):
        self.assertEqual(find_ffmpeg(), '/opt/homebrew/bin/ffmpeg')


if __name__ == '__main__':
    unittest.main()
