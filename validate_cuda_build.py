"""Compile every generated CUDA kernel with NVRTC, without requiring a GPU.

This checks CUDA C++ syntax and architecture support, not kernel execution.
"""
import ctypes as C
import importlib.metadata as md
import os
from pathlib import Path
from creative_hdr import CreativeHDRSettings
from cuda_colour import cuda_source


def main():
    import cupy
    assert callable(cupy.cuda.driver.ctxGetCurrent)
    dist = md.distribution('nvidia-cuda-nvrtc-cu12')
    dlls = [Path(dist.locate_file(p)) for p in dist.files or []
            if p.name.startswith('nvrtc64_') and p.suffix.lower() == '.dll' and '.alt.' not in p.name]
    if len(dlls) != 1:
        raise RuntimeError('Cannot identify primary NVRTC runtime: ' + repr(dlls))
    handles = [os.add_dll_directory(str(dlls[0].parent))]
    os.environ['PATH'] = str(dlls[0].parent) + os.pathsep + os.environ.get('PATH', '')
    builtins = [Path(dist.locate_file(p)) for p in dist.files or []
                if p.name.startswith('nvrtc-builtins64_') and p.suffix.lower() == '.dll' and '.alt.' not in p.name]
    if len(builtins) != 1:
        raise RuntimeError('Cannot identify NVRTC builtins: ' + repr(builtins))
    builtins_handle = C.WinDLL(str(builtins[0]))
    print('Primary NVRTC:', dlls[0], 'builtins:', builtins[0])
    nvrtc = C.WinDLL(str(dlls[0]))
    nvrtc.nvrtcCreateProgram.argtypes = [C.POINTER(C.c_void_p), C.c_char_p, C.c_char_p,
                                        C.c_int, C.c_void_p, C.c_void_p]
    nvrtc.nvrtcCompileProgram.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_char_p)]
    nvrtc.nvrtcGetProgramLogSize.argtypes = [C.c_void_p, C.POINTER(C.c_size_t)]
    nvrtc.nvrtcGetProgramLog.argtypes = [C.c_void_p, C.c_char_p]
    nvrtc.nvrtcDestroyProgram.argtypes = [C.POINTER(C.c_void_p)]
    runtime = md.distribution('nvidia-cuda-runtime-cu12')
    includes = {str(Path(runtime.locate_file(p)).parent) for p in runtime.files or []
                if p.name == 'cuda_runtime.h'}
    options = ['--std=c++11', '--fmad=false', '--gpu-architecture=compute_120']
    options += ['--include-path=' + p for p in sorted(includes)]
    opt = (C.c_char_p * len(options))(*(x.encode() for x in options))
    grades = [CreativeHDRSettings(), CreativeHDRSettings(enabled=True),
              CreativeHDRSettings(enabled=True, peak_nits=4000, exposure=.8,
                                  contrast=1.3, midtones=.4, shadows=-.5,
                                  black_floor=.1, saturation=1.2, vibrance=.4,
                                  temperature=.2, tint=-.3)]
    for i, grade in enumerate(grades):
        program = C.c_void_p()
        rc = nvrtc.nvrtcCreateProgram(C.byref(program), cuda_source(grade).encode(),
                                     b'hdr_colour.cu', 0, None, None)
        if rc:
            raise RuntimeError('nvrtcCreateProgram error ' + str(rc))
        try:
            rc = nvrtc.nvrtcCompileProgram(program, len(options), opt)
            length = C.c_size_t()
            nvrtc.nvrtcGetProgramLogSize(program, C.byref(length))
            text = C.create_string_buffer(max(1, length.value))
            nvrtc.nvrtcGetProgramLog(program, text)
            print(text.value.decode(errors='replace'))
            if rc:
                raise RuntimeError('NVRTC compile failed: ' + str(rc))
            print('CUDA kernel configuration', i, 'compiled for compute_120; execution not tested.')
        finally:
            nvrtc.nvrtcDestroyProgram(C.byref(program))
    for handle in handles:
        handle.close()


if __name__ == '__main__':
    main()
