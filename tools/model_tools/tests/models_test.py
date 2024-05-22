import unittest
import os

import omz_tools.models as omz


class TestOMZModel(unittest.TestCase):
    def test_load_intel(self):
        face_detection = omz.OMZModel.download('face-detection-0200', precision='FP16-INT8', download_dir='models')
        xml_path = face_detection.model_path

        self.assertTrue(os.path.exists(xml_path))

    def test_load_public(self):
        model = omz.OMZModel.download('colorization-v2', precision='FP32')
        xml_path = model.model_path

        self.assertTrue(os.path.exists(xml_path))

    def test_load_from_pretrained(self):
        model = omz.OMZModel.from_pretrained('models/intel/face-detection-0200/FP16-INT8/face-detection-0200.xml')
        xml_path = model.model_path

        self.assertTrue(os.path.exists(xml_path))

    def test_get_accuracy_checker_config(self):
        model = omz.OMZModel.download('colorization-v2', cache_dir='models/public/colorization-v2/')
        self.assertIsInstance(model.accuracy_checker_config(), dict)

    def test_get_model_config(self):
        model = omz.OMZModel.download('colorization-v2', cache_dir='models/public/colorization-v2/')
        self.assertIsInstance(model.model_config(), dict)

    def test_vocab_loading(self):
        model = omz.OMZModel.download('bert-large-uncased-whole-word-masking-squad-0001', precision='FP16')
        self.assertIsInstance(model.vocab(), dict)

    def test_load_public_composite(self):
        model = omz.OMZModel.download('wavernn-rnn', precision='FP32')
        xml_path = model.model_path

        self.assertTrue(os.path.exists(xml_path))

    def test_preferable_input_shape(self):
        model = omz.OMZModel.download('colorization-v2', cache_dir='models/public/colorization-v2/')
        self.assertEqual(model.preferable_input_shape('data_l'), [1, 1, 256, 256])

    def test_input_layout(self):
        model = omz.OMZModel.download('colorization-v2', cache_dir='models/public/colorization-v2/')
        self.assertEqual(model.layout('data_l'), 'NCHW')

if __name__ == '__main__':
    unittest.main()
