"""Build the Windows CUDA edition as a folder, including optional GPU libraries.

No NVIDIA driver is bundled. FFmpeg/FFprobe are still discovered on PATH.
A folder build avoids unpacking large CUDA runtimes at every application start.
"""
import importlib.metadata as metadata
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    if sys.platform != 'win32':
        raise SystemExit('Build this Windows executable on Windows.')
    name = 'HDRVideoStudio-GPU'
    command = [sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean', '--onedir',
               '--windowed', '--name', name, '--collect-all', 'cupy',
               '--collect-all', 'cupy_backends', '--hidden-import', 'fastrlock',
               '--hidden-import', 'PyNvVideoCodec', '--hidden-import', 'PyNvVideoCodec.decoders',
               '--hidden-import', 'PyNvVideoCodec.decoders.SimpleDecoder',
               '--hidden-import', 'PyNvVideoCodec.decoders.ThreadedDecoder', 'gpu_only_studio.py']
    subprocess.run(command, check=True)
    root = Path('dist') / name
    internal = root / '_internal'
    nvc = metadata.distribution('PyNvVideoCodec')
    # PyNvVideoCodec dynamically picks a versioned .pyd based on the driver.
    # Preserve its entire package layout, rather than guessing one extension.
    shutil.copytree(nvc.locate_file('PyNvVideoCodec'), internal / 'PyNvVideoCodec', dirs_exist_ok=True)
    cuda = internal / 'cuda'
    (cuda / 'bin').mkdir(parents=True, exist_ok=True)
    (cuda / 'include').mkdir(exist_ok=True)
    licenses = root / 'licenses'
    licenses.mkdir(exist_ok=True)
    for package in ('nvidia-cuda-runtime-cu12', 'nvidia-cuda-nvrtc-cu12'):
        dist = metadata.distribution(package)
        for item in dist.files or []:
            path = Path(dist.locate_file(item))
            parts = item.parts
            if path.suffix.lower() == '.dll':
                shutil.copy2(path, cuda / 'bin' / path.name)
            elif 'include' in parts and path.is_file():
                target = cuda / 'include' / Path(*parts[parts.index('include')+1:])
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    # Keep redistribution notices with the packaged dependencies.
    for package in ('PyNvVideoCodec', 'cupy-cuda12x', 'nvidia-cuda-runtime-cu12', 'nvidia-cuda-nvrtc-cu12'):
        dist = metadata.distribution(package)
        for item in dist.files or []:
            path = Path(dist.locate_file(item))
            if path.is_file() and any(x in path.name.lower() for x in ('license', 'notice', 'eula')):
                shutil.copy2(path, licenses / (package + '-' + path.name))
    shutil.copy2('GPU_ONLY_MODE.md', root / 'GPU_ONLY_MODE.md')
    (root / 'BUILD_INFO.txt').write_text(
        subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True) + '\n' +
        '\n'.join(package + '==' + metadata.version(package) for package in (
            'PyNvVideoCodec', 'cupy-cuda12x', 'nvidia-cuda-runtime-cu12', 'nvidia-cuda-nvrtc-cu12')),
        encoding='utf-8')
    # Do not distribute incidental font files copied by dependency hooks.
    for font in root.rglob('*'):
        if font.is_file() and font.suffix.lower() in ('.ttf', '.ttc', '.otf', '.woff', '.woff2', '.eot', '.pfb', '.pfa', '.fon', '.fnt'):
            font.unlink()
    print('Built', root.resolve())


if __name__ == '__main__':
    main()
