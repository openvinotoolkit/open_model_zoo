import numpy as np
import unittest
import os

import openvino.model_zoo.open_model_zoo as omz

from openvino.inference_engine import IECore

class TestTopologies(unittest.TestCase):
    def test_load_intel(self):
        face_detection = omz.Model.download('face-detection-0200', precision='FP16-INT8')
        xml_path = face_detection.model_path

        self.assertTrue(os.path.exists(xml_path))

        ie = IECore()
        net = ie.read_network(xml_path)

        input_name = next(iter(net.input_info))
        output_name = next(iter(net.outputs))

        self.assertEqual(net.input_info[input_name].input_data.shape, [1, 3, 256, 256])
        self.assertEqual(net.outputs[output_name].shape, [1, 1, 200, 7])

    def test_load_public(self):
        model = omz.Model.download('colorization-v2', precision='FP32')
        xml_path = model.model_path

        self.assertTrue(os.path.exists(xml_path))

        ie = IECore()
        net = ie.read_network(xml_path)

        input_name = next(iter(net.input_info))
        output_name = next(iter(net.outputs))

        self.assertEqual(net.input_info[input_name].input_data.shape, [1, 1, 256, 256])
        self.assertEqual(net.outputs[output_name].shape, [1, 2, 256, 256])

    def test_load_from_pretrained(self):
        model = omz.Model.from_pretrained('models/public/colorization-v2/FP32/colorization-v2.xml')
        xml_path = model.model_path

        self.assertTrue(os.path.exists(xml_path))

        ie = IECore()
        net = ie.read_network(xml_path)

        input_name = next(iter(net.input_info))
        output_name = next(iter(net.outputs))

        self.assertEqual(net.input_info[input_name].input_data.shape, [1, 1, 256, 256])
        self.assertEqual(net.outputs[output_name].shape, [1, 2, 256, 256])

    def test_get_accuracy_checker_config(self):
        model = omz.Model.download('colorization-v2', cache_dir='models/public/colorization-v2/')
        self.assertIsInstance(model.accuracy_checker_config, dict)

    def test_get_model_config(self):
        model = omz.Model.download('colorization-v2', cache_dir='models/public/colorization-v2/')
        self.assertIsInstance(model.model_config, dict)

    def test_infer_model(self):
        model = omz.Model.download('colorization-v2', cache_dir='models/public/colorization-v2/')

        ie = IECore()
        net = ie.read_network(model.model_path)
        input_name = next(iter(net.input_info))
        output_name = next(iter(net.outputs))

        inputs = {input_name: np.zeros((1, 1, 256, 256))}
        output = model(inputs, ie)
        self.assertEqual(output[output_name].shape, (1, 2, 256, 256))

    def test_load_public_composite(self):
        model = omz.Model.download('mtcnn-p', precision='FP32')
        xml_path = model.model_path

        self.assertTrue(os.path.exists(xml_path))

        ie = IECore()
        net = ie.read_network(xml_path)

        input_name = next(iter(net.input_info))
        expected_shapes = {
            'conv4-2': [1, 4, 355, 635],
            'prob1': [1, 2, 355, 635]
        }

        self.assertEqual(net.input_info[input_name].input_data.shape, [1, 3, 720, 1280])
        for name, value in net.outputs.items():
            self.assertEqual(expected_shapes[name], value.shape)

    def test_input_shape(self):
        model = omz.Model.download('colorization-v2', cache_dir='models/public/colorization-v2/')
        self.assertEqual(model.input_shape('data_l'), [1, 1, 256, 256])

    def test_input_layout(self):
        model = omz.Model.download('colorization-v2', cache_dir='models/public/colorization-v2/')
        self.assertEqual(model.layout('data_l'), 'NCHW')

if __name__ == '__main__':
    unittest.main()
