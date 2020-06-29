from copy import deepcopy
from collections import namedtuple
import warnings
try:
    from openvino.inference_engine import ResizeAlgorithm, PreProcessInfo, ColorFormat
except ImportError:
    ResizeAlgorithm, PreProcessInfo, ColorFormat = None, None, None


def ie_preprocess_available():
    return PreProcessInfo is not None


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


def get_color_format_op(config):
    source_color = config['type'].split('_')[0].upper()
    try:
        ie_color_format = ColorFormat[source_color]
    except KeyError:
        return None
    return PreprocessingOp('color_format', ie_color_format)


SUPPORTED_PREPROCESSING_OPS = {
    'resize': get_resize_op,
    'auto_resize': get_resize_op,
    'bgr_to_rgb': get_color_format_op,
    'rgb_to_bgr': get_color_format_op,
    'nv12_to_bgr': get_color_format_op,
    'nv12_to_rgb': get_color_format_op
}


class IEPreprocessor:
    def __init__(self, config):
        self.config = config
        self.configure()

    def configure(self):
        steps = []
        step_names = set()
        keep_preprocessing_config = deepcopy(self.config)
        for preprocessor in reversed(self.config):
            if preprocessor['type'] not in SUPPORTED_PREPROCESSING_OPS:
                break
            ie_ops = self.get_op(preprocessor)
            if not ie_ops:
                break
            if ie_ops.name in step_names:
                break
            step_names.add(ie_ops.name)
            steps.append(ie_ops)
            keep_preprocessing_config = keep_preprocessing_config[:-1]
        self.steps = steps
        self.keep_preprocessing_info = keep_preprocessing_config
        if not steps:
            warnings.warn('no preprocessing steps for transition to PreProcessInfo')

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
