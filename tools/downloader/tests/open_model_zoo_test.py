import unittest
import os

import open_model_zoo.open_model_zoo as omz

from openvino.inference_engine import IECore

class TestTopologies(unittest.TestCase):
    def test_load_intel(self):
        face_detection = omz.Model('face-detection-0200', 'FP16-INT8')
        xml_path, bin_path = face_detection.xml_path, face_detection.bin_path

        self.assertTrue(os.path.exists(xml_path))
        self.assertTrue(os.path.exists(bin_path))

        ie = IECore()
        net = ie.read_network(xml_path, bin_path)
        exec_net = ie.load_network(net, 'CPU')
        infer_request = exec_net.requests[0]

        input_blob = infer_request.input_blobs[next(iter(net.input_info))]
        output_blob = infer_request.output_blobs[next(iter(net.outputs))]

        self.assertEqual(input_blob.tensor_desc.dims, [1, 3, 256, 256])
        self.assertEqual(output_blob.tensor_desc.dims, [1, 1, 200, 7])

    def test_load_public(self):
        ssd = omz.Model('mobilenet-ssd', 'FP32')
        xml_path, bin_path = ssd.xml_path, ssd.bin_path

        self.assertTrue(os.path.exists(xml_path))
        self.assertTrue(os.path.exists(bin_path))

        ie = IECore()
        net = ie.read_network(xml_path, bin_path)
        exec_net = ie.load_network(net, 'CPU')
        infer_request = exec_net.requests[0]

        input_blob = infer_request.input_blobs[next(iter(net.input_info))]
        output_blob = infer_request.output_blobs[next(iter(net.outputs))]

        self.assertEqual(input_blob.tensor_desc.dims, [1, 3, 300, 300])
        self.assertEqual(output_blob.tensor_desc.dims, [1, 1, 100, 7])

if __name__ == '__main__':
    unittest.main()
