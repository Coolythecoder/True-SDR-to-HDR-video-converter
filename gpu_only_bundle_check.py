"""Driver-free checks run INSIDE the packaged Windows executable.

Imports optional CuPy binary dependencies and compiles the shipped CUDA source
with the shipped NVRTC libraries. Does not claim GPU/codec execution.
"""
import ctypes as C
import json
import os
from pathlib import Path
import sys
import traceback
from creative_hdr import CreativeHDRSettings
from cuda_colour import cuda_source


def check():
    root = Path(sys._MEIPASS)
    cuda = root / 'cuda'
    os.environ['CUDA_PATH'] = str(cuda)
    os.environ['PATH'] = str(cuda / 'bin') + os.pathsep + os.environ.get('PATH', '')
    handles = [os.add_dll_directory(str(cuda / 'bin'))]
    builtin = C.WinDLL(str(cuda / 'bin/nvrtc-builtins64_128.dll'))
    nvrtc = C.WinDLL(str(cuda / 'bin/nvrtc64_120_0.dll'))
    import cupy
    from fastrlock.rlock import FastRLock
    assert callable(cupy.cuda.driver.ctxGetCurrent)
    assert FastRLock() is not None
    assert (root / 'PyNvVideoCodec/VersionCheck.cp311-win_amd64.pyd').is_file()
    for version in ('121','130'):
        assert (root / f'PyNvVideoCodec/PyNvVideoCodec_{version}.cp311-win_amd64.pyd').is_file()
    nvrtc.nvrtcCreateProgram.argtypes = [C.POINTER(C.c_void_p),C.c_char_p,C.c_char_p,C.c_int,C.c_void_p,C.c_void_p]
    nvrtc.nvrtcCompileProgram.argtypes = [C.c_void_p,C.c_int,C.POINTER(C.c_char_p)]
    nvrtc.nvrtcGetProgramLogSize.argtypes = [C.c_void_p,C.POINTER(C.c_size_t)]
    nvrtc.nvrtcGetProgramLog.argtypes = [C.c_void_p,C.c_char_p]
    nvrtc.nvrtcDestroyProgram.argtypes = [C.POINTER(C.c_void_p)]
    options = [b'--std=c++11',b'--fmad=false',b'--gpu-architecture=compute_120',
               ('--include-path='+str(cuda/'include')).encode()]
    opts = (C.c_char_p*len(options))(*options)
    for grade in (CreativeHDRSettings(),CreativeHDRSettings(enabled=True),
                  CreativeHDRSettings(enabled=True,peak_nits=4000,exposure=.8,contrast=1.3,
                                      midtones=.4,shadows=-.5,black_floor=.1,saturation=1.2,
                                      vibrance=.4,temperature=.2,tint=-.3)):
        program = C.c_void_p()
        rc = nvrtc.nvrtcCreateProgram(C.byref(program),cuda_source(grade).encode(),b'bundled.cu',0,None,None)
        if rc:
            raise RuntimeError('Bundled NVRTC create failed: '+str(rc))
        try:
            rc = nvrtc.nvrtcCompileProgram(program,len(options),opts)
            length = C.c_size_t()
            nvrtc.nvrtcGetProgramLogSize(program,C.byref(length))
            text = C.create_string_buffer(max(1,length.value))
            nvrtc.nvrtcGetProgramLog(program,text)
            if rc:
                raise RuntimeError(text.value.decode(errors='replace'))
        finally:
            nvrtc.nvrtcDestroyProgram(C.byref(program))
    return dict(success=True,cupy=cupy.__version__,fastrlock_import=True,
                bundled_nvrtc_configurations_compiled=3,architecture='compute_120',
                hardware_execution_tested=False)


def main():
    output = Path(sys.argv[sys.argv.index('--bundle-check')+1])
    try:
        result = check()
    except Exception:
        result = dict(success=False,error=traceback.format_exc(),hardware_execution_tested=False)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    return 0 if result['success'] else 1
