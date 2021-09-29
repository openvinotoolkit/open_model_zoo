import numpy as np
import unittest
import os

import open_model_zoo.open_model_zoo as omz

from openvino.inference_engine import IECore

class TestTopologies(unittest.TestCase):
    def test_load_intel(self):
        face_detection = omz.Model('face-detection-0200', precision='FP16-INT8')
        xml_path, bin_path = face_detection.model_path, face_detection.bin_path

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
        model = omz.Model('colorization-v2', precision='FP32')
        xml_path, bin_path = model.model_path, model.bin_path

        self.assertTrue(os.path.exists(xml_path))
        self.assertTrue(os.path.exists(bin_path))

        ie = IECore()
        net = ie.read_network(xml_path, bin_path)
        exec_net = ie.load_network(net, 'CPU')
        infer_request = exec_net.requests[0]

        input_blob = infer_request.input_blobs[next(iter(net.input_info))]
        output_blob = infer_request.output_blobs[next(iter(net.outputs))]

        self.assertEqual(input_blob.tensor_desc.dims, [1, 1, 256, 256])
        self.assertEqual(output_blob.tensor_desc.dims, [1, 2, 256, 256])

    def test_load_from_prtained(self):
        model = omz.Model.from_pretrained('models/public/colorization-v2/FP32/colorization-v2.xml')
        xml_path, bin_path = model.model_path, model.bin_path

        self.assertTrue(os.path.exists(xml_path))
        self.assertTrue(os.path.exists(bin_path))

        ie = IECore()
        net = ie.read_network(xml_path, bin_path)
        exec_net = ie.load_network(net, 'CPU')
        infer_request = exec_net.requests[0]

        input_blob = infer_request.input_blobs[next(iter(net.input_info))]
        output_blob = infer_request.output_blobs[next(iter(net.outputs))]

        self.assertEqual(input_blob.tensor_desc.dims, [1, 1, 256, 256])
        self.assertEqual(output_blob.tensor_desc.dims, [1, 2, 256, 256])

    def test_load_model(self):
        model = omz.Model('colorization-v2', cache_dir='models/public/colorization-v2/')
        ie = IECore()
        model.load(ie)
        self.assertIsNotNone(model.net)
        self.assertIsNotNone(model.exec_net)

    def test_infer_model(self):
        model = omz.Model('colorization-v2', cache_dir='models/public/colorization-v2/')
        # dummy_inputs = np.zeros(input_shape)
        # ie = IECore()
        


if __name__ == '__main__':
    unittest.main()
