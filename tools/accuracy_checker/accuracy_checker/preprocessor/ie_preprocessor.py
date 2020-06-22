from copy import deepcopy
from collections import namedtuple
import warnings
from openvino.inference_engine import ResizeAlgorithm, PreProcessInfo

PreprocessingOp = namedtuple('PreprocessingOp', ['name', 'value'])

def get_resize_op(config):
    supported_interpolations = {
        'LINEAR': ResizeAlgorithm.RESIZE_BILINEAR,
        'BILINEAR': ResizeAlgorithm.RESIZE_BILINEAR,
        'AREA': ResizeAlgorithm.RESIZE_AREA
    }
    if 'aspect_ratio' in config:
        return None
    interpolation = config.get('interpolation', 'BILINEAR').upper()
    if interpolation not in supported_interpolations:
        return None
    return PreprocessingOp('resize_algorithm', supported_interpolations[interpolation])

SUPPORTED_PREPROCESSING_OPS = {'resize': get_resize_op}

class IEPreprocessor:
    def __init__(self, config):
        self.config = config
        self.configure()

    def configure(self):
        steps = []
        keep_preprocessing_config = deepcopy(self.config)
        if self.config and self.config[-1]['type'] in SUPPORTED_PREPROCESSING_OPS:
            ie_ops = self.get_op(self.config[-1])
            if ie_ops:
                steps.append(ie_ops)
                keep_preprocessing_config = self.config[:-1]
        self.steps = steps
        self.keep_preprocessing_info = keep_preprocessing_config
        if not steps:
            warnings.warn('no preprocessing steps for transition to PreprocessingInfo')

    @staticmethod
    def get_op(preprocessing_config):
        preprocessing_getter = SUPPORTED_PREPROCESSING_OPS[preprocessing_config['type']]
        return preprocessing_getter(preprocessing_config)

    @property
    def preprocess_info(self):
        preprocess_info = PreProcessInfo()
        for (name, value) in self.steps:
            setattr(preprocess_info, name, value)
        return preprocess_info
