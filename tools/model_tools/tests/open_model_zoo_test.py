import numpy as np
import unittest
import os

import openvino.model_zoo.open_model_zoo as omz

from openvino.model_zoo.model_api.models import Classification

from openvino.runtime import Core

class TestOMZModel(unittest.TestCase):
    def test_load_intel(self):
        face_detection = omz.OMZModel.download('face-detection-0200', precision='FP16-INT8')
        xml_path = face_detection.model_path

        self.assertTrue(os.path.exists(xml_path))

        ie = Core()
        ie_model = ie.read_model(xml_path)

        self.assertEqual(list(ie_model.inputs[0].shape), [1, 3, 256, 256])
        self.assertEqual(list(ie_model.outputs[0].shape), [1, 1, 200, 7])

    def test_load_public(self):
        model = omz.OMZModel.download('colorization-v2', precision='FP32')
        xml_path = model.model_path

        self.assertTrue(os.path.exists(xml_path))

        ie = Core()
        ie_model = ie.read_model(xml_path)

        self.assertEqual(list(ie_model.inputs[0].shape), [1, 1, 256, 256])
        self.assertEqual(list(ie_model.outputs[0].shape), [1, 2, 256, 256])

    def test_load_from_pretrained(self):
        model = omz.OMZModel.from_pretrained('models/intel/face-detection-0200/FP16-INT8/face-detection-0200.xml')
        xml_path = model.model_path

        self.assertTrue(os.path.exists(xml_path))

        ie = Core()
        ie_model = ie.read_model(xml_path)

        self.assertEqual(list(ie_model.inputs[0].shape), [1, 3, 256, 256])
        self.assertEqual(list(ie_model.outputs[0].shape), [1, 1, 200, 7])

    def test_get_accuracy_checker_config(self):
        model = omz.OMZModel.download('colorization-v2', cache_dir='models/public/colorization-v2/')
        self.assertIsInstance(model.accuracy_checker_config(), dict)

    def test_get_model_config(self):
        model = omz.OMZModel.download('colorization-v2', cache_dir='models/public/colorization-v2/')
        self.assertIsInstance(model.model_config(), dict)

    def test_infer_model(self):
        ie = Core()
        model = omz.OMZModel.download('colorization-v2', cache_dir='models/public/colorization-v2/', ie=ie)

        ie_model = ie.read_model(model.model_path)
        input_name = ie_model.inputs[0].get_any_name()

        inputs = {input_name: np.zeros((1, 1, 256, 256))}
        output = next(iter(model(inputs).values()))
        self.assertEqual(output.shape, (1, 2, 256, 256))

    def test_infer_non_vision_model(self):
        ie = Core()
        model = omz.OMZModel.download('bert-large-uncased-whole-word-masking-squad-0001', precision='FP16', ie=ie)

        ie_model = ie.read_model(model.model_path)
        input_names = [input.get_any_name() for input in ie_model.inputs]
        input_shapes = [input.shape for input in ie_model.inputs]

        expected_shapes = {
            'output_s': (1, 384),
            'output_e': (1, 384)
        }

        inputs = {}
        for name, shape in zip(input_names, input_shapes):
            inputs[name] = np.zeros(list(shape))

        outputs = model(inputs)
        for name, output in outputs.items():
            self.assertEqual(output.shape, expected_shapes[name.get_any_name()])

    def test_load_public_composite(self):
        model = omz.OMZModel.download('mtcnn-p', precision='FP32')
        xml_path = model.model_path

        self.assertTrue(os.path.exists(xml_path))

        ie = Core()
        ie_model = ie.read_model(xml_path)

        expected_shapes = {
            'conv4-2': [1, 4, 355, 635],
            'prob1': [1, 2, 355, 635]
        }

        self.assertEqual(list(ie_model.inputs[0].shape), [1, 3, 720, 1280])
        for output in ie_model.outputs:
            output_name = output.get_any_name()
            self.assertEqual(list(output.shape), expected_shapes[output_name])

    def test_preferable_input_shape(self):
        model = omz.OMZModel.download('colorization-v2', cache_dir='models/public/colorization-v2/')
        self.assertEqual(model.preferable_input_shape('data_l'), [1, 1, 256, 256])

    def test_input_layout(self):
        model = omz.OMZModel.download('colorization-v2', cache_dir='models/public/colorization-v2/')
        self.assertEqual(model.layout('data_l'), 'NCHW')

    def test_model_api_inference(self):
        ie = Core()
        model = omz.OMZModel.download('densenet-121', cache_dir='models/public/densenet-121/', ie=ie)

        input = np.zeros((224, 224, 3))
        result = model.model_api_inference(input, model_creator=Classification, configuration={'topk': 1})

        self.assertEqual(len(result), 1)

    def test_model_api_auto_model_creation(self):
        ie = Core()
        model = omz.OMZModel.download('pspnet-pytorch', cache_dir='models/public/pspnet-pytorch/', ie=ie)

        input = np.random.randint(0, 256, (512, 512, 3))
        result = model.model_api_inference(input)

        self.assertEqual(result.shape, (512, 512))

    def test_vocab_loading(self):
        model = omz.OMZModel.download('bert-large-uncased-whole-word-masking-squad-0001', precision='FP16')
        self.assertIsInstance(model.vocab(), dict)

if __name__ == '__main__':
    unittest.main()
