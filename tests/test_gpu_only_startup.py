"""Native startup wiring; mocked codecs do not claim GPU execution."""
from unittest import TestCase, mock
from fractions import Fraction
import numpy as np
import gpu_only_export as gpu


class NativeStartupTests(TestCase):
    def test_probe_geometry_context_and_hdr_metadata_reach_native_encoder(self):
        info = gpu.SourceInfo(3840,2160,10,Fraction(144),Fraction(1,14400),False,False)
        cp, nvc, stream = mock.MagicMock(), mock.MagicMock(), mock.MagicMock()
        cp.asarray.side_effect = lambda x:x
        cp.cuda.driver.ctxGetCurrent.return_value = 321
        stream.ptr = 123
        nvc.CreateEncoder.return_value.Encode.return_value = b'packet'
        nvc.CreateEncoder.return_value.EndEncode.return_value = b''
        with mock.patch.object(gpu,'CUDAProcessor') as processor, \
             mock.patch.object(gpu,'_run',return_value=np.zeros((3,256,256),dtype='<f4').tobytes()), \
             mock.patch.object(gpu,'verify_output') as verified:
            processor.return_value.process.return_value.get.return_value = np.zeros((3,256,256),dtype='<f4')
            gpu.validate_cuda_backend(cp,nvc,'ffmpeg','ffprobe',info,gpu.GPUOnlySettings(),stream,
                                      lambda:False,lambda p:None,lambda text:None)
            args,kw=nvc.CreateEncoder.call_args
            self.assertEqual(args,(256,256,'P010',False))
            self.assertEqual(kw['cudacontext'],321)
            self.assertEqual(kw['cudastream'],123)
            self.assertEqual(kw['fps'],144)
            self.assertEqual(processor.call_args.args[1].width,256)
            verified.assert_called_once()
            self.assertEqual(verified.call_args.args[3:6],(2,100,50))
            self.assertEqual(verified.call_args.kwargs['timeout'],120)
            sei=nvc.CreateEncoder.return_value.Encode.call_args.args[2]
            self.assertEqual([m[0]['sei_type'] for m in sei],[137,144])
