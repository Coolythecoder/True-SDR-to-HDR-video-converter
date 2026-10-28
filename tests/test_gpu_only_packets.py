"""Native NVENC packet batches are serialized without dropping flush output."""
import io
import unittest

import numpy as np

from gpu_only_export import _write_encoded_packets


class EncodedPacketTests(unittest.TestCase):
    def test_buffered_encode_and_multi_packet_flush_preserve_every_byte(self):
        output = io.BytesIO()
        _write_encoded_packets(output, [])
        self.assertEqual(output.getvalue(), b'')
        _write_encoded_packets(output, [
            {'data': b'\x00\x00\x00\x01\x26\xff', 'timestamp': 0, 'picture_type': 3},
        ])
        _write_encoded_packets(output, [
            {'data': b'\x00\x00\x00\x01\x02\x80', 'timestamp': 1, 'picture_type': 0},
            {'data': b'\x00\x00\x00\x01\x02\x81', 'timestamp': 2, 'picture_type': 0},
        ])
        self.assertEqual(output.getvalue(),
                         b'\x00\x00\x00\x01\x26\xff'
                         b'\x00\x00\x00\x01\x02\x80'
                         b'\x00\x00\x00\x01\x02\x81')

    def test_direct_bitstream_results_remain_supported(self):
        data = b'\x00\x00\x01\xff'
        for packets in (data, bytearray(data), memoryview(data),
                        np.frombuffer(data, dtype=np.uint8), b''):
            with self.subTest(type=type(packets), length=len(packets)):
                output = io.BytesIO()
                _write_encoded_packets(output, packets)
                self.assertEqual(output.getvalue(), bytes(packets))

    def test_malformed_packet_does_not_silently_disappear(self):
        with self.assertRaises(KeyError):
            _write_encoded_packets(io.BytesIO(), [{'timestamp': 0}])


if __name__ == '__main__':
    unittest.main()
