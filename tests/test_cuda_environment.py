"""Windows CUDA discovery without importing CuPy or requiring a GPU."""
import ctypes
from importlib import metadata
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import gpu_only_export as gpu


@unittest.skipUnless(os.name == 'nt', 'Windows DLL discovery')
class CUDAEnvironmentTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.runtime = self.root / 'user-site/nvidia/cuda_runtime'
        self.nvrtc = self.root / 'user-site/nvidia/cuda_nvrtc'
        for path in (self.runtime, self.nvrtc):
            (path / 'bin').mkdir(parents=True)
        (self.nvrtc / 'bin/nvrtc-builtins64_128.dll').touch()
        (self.nvrtc / 'bin/nvrtc-builtins64_128.alt.dll').touch()

        def distribution(name):
            if name not in ('nvidia-cuda-runtime-cu12', 'nvidia-cuda-nvrtc-cu12'):
                raise metadata.PackageNotFoundError(name)
            result = mock.Mock()
            result.locate_file.side_effect = lambda relative: self.root / 'user-site' / relative
            return result

        patches = [
            mock.patch.dict(os.environ, {'PATH': 'existing-path'}, clear=True),
            mock.patch.object(sys, '_MEIPASS', str(self.root / 'app'), create=True),
            mock.patch.object(metadata, 'distribution', side_effect=distribution),
            mock.patch.object(gpu.shutil, 'which', return_value=None),
            mock.patch.object(os, 'add_dll_directory'),
            mock.patch.object(ctypes, 'WinDLL'),
            mock.patch.object(gpu, '_DLL_HANDLES', []),
        ]
        mocks = []
        for patch in patches:
            mocks.append(patch.start())
            self.addCleanup(patch.stop)
        self.distribution, self.which, self.add_dll, self.load_dll = mocks[2:6]

    def test_pip_runtime_is_discovered_outside_python_prefix(self):
        gpu._configure_cuda_environment()
        self.assertEqual(os.environ['CUDA_PATH'], str(self.runtime))
        self.assertIn(str(self.runtime / 'bin'), os.environ['PATH'].split(os.pathsep))
        self.assertIn(str(self.nvrtc / 'bin'), os.environ['PATH'].split(os.pathsep))
        self.load_dll.assert_called_once_with(str(self.nvrtc / 'bin/nvrtc-builtins64_128.dll'))

    def test_existing_toolkit_is_preserved(self):
        toolkit = self.root / 'toolkit'
        (toolkit / 'bin').mkdir(parents=True)
        os.environ['CUDA_PATH'] = str(toolkit)
        gpu._configure_cuda_environment()
        self.assertEqual(os.environ['CUDA_PATH'], str(toolkit))

    def test_stale_cuda_path_falls_back_to_runtime(self):
        os.environ['CUDA_PATH'] = str(self.root / 'missing')
        gpu._configure_cuda_environment()
        self.assertEqual(os.environ['CUDA_PATH'], str(self.runtime))

    def test_nvcc_toolkit_precedes_pip_runtime(self):
        toolkit = self.root / 'toolkit'
        (toolkit / 'bin').mkdir(parents=True)
        self.which.return_value = str(toolkit / 'bin/nvcc.exe')
        gpu._configure_cuda_environment()
        self.assertEqual(os.environ['CUDA_PATH'], str(toolkit))

    def test_frozen_bundle_needs_no_distribution_metadata(self):
        bundled = self.root / 'app/cuda'
        (bundled / 'bin').mkdir(parents=True)
        os.environ['CUDA_PATH'] = str(self.runtime)
        gpu._configure_cuda_environment()
        self.assertEqual(os.environ['CUDA_PATH'], str(bundled))
        self.distribution.assert_not_called()
        self.add_dll.assert_called_once_with(str(bundled / 'bin'))

    def test_missing_optional_wheels_do_not_invent_cuda_root(self):
        self.distribution.side_effect = metadata.PackageNotFoundError
        gpu._configure_cuda_environment()
        self.assertNotIn('CUDA_PATH', os.environ)
        self.add_dll.assert_not_called()


if __name__ == '__main__':
    unittest.main()
