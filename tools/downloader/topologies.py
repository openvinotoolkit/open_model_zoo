import sys
import os
import re
import yaml
import types

from pathlib import Path

import common

if 'INTEL_OPENVINO_DIR' in os.environ:  # As a part of OpenVINO
    _loc = Path(os.environ['INTEL_OPENVINO_DIR']) / 'deployment_tools' / 'open_model_zoo'
    _models_dir = _loc / 'models'
    _downloader = _loc / 'tools' / 'downloader' / 'downloader.py'
else:  # Standalone
    _loc = Path(__file__).parent
    _models_dir = _loc / '../../models'
    _downloader = _loc / 'downloader.py'

_precision_marker = '$precision'
_cache = Path(os.getenv('OPENCV_OPEN_MODEL_ZOO_CACHE_DIR', 'models')).resolve()


class _args: config = None
for _topology in common.load_topologies(_args):
    _name = _topology.name.replace('-', '_').replace('.', '_')
    _files = _topology.files
    _download_dir = _cache / _topology.subdirectory
    _mo_args = {}
    if _topology.mo_args:
        for _arg in _topology.mo_args:
            _tokens = _arg.split('=')
            _mo_args[_tokens[0]] = _tokens[1] if len(_tokens) == 2 else None

    # Rsolve weights and text files names.
    assert(len(_files) > 0)

    _config_path = str(_files[0].name)
    _model_path = str(_files[-1].name)

    if len(_files) > 2:
        assert(_topology.framework == 'dldt'), ('Unexpected framework type: ' + _framework)
        # Get only basename and add precision marker
        _pos = _config_path.find('/')
        if _config_path[:_pos] in common.KNOWN_PRECISIONS:
            _config_path = _precision_marker / _config_path[_pos:]

        _pos = _model_path.find('/')
        if _model_path[:_pos] in common.KNOWN_PRECISIONS:
            _model_path = _precision_marker / _model_path[_pos:]

    # To manage origin files location from archives
    if '--input_model' in _mo_args:
        _model_path = _mo_args['--input_model'].replace('$dl_dir/', '')
        _config_path = ''  # If there is a text file - we will set it next

    for key in ['--input_proto', '--input_symbol']:
        if key in _mo_args:
            _config_path = _mo_args[key].replace('$dl_dir/', '')

    _model_path = (_download_dir / _model_path).resolve()
    if _config_path:
        _config_path = (_download_dir / _config_path).resolve()


    exec("""
class {className}:
    '''
    {description}

    License: {license}

    Attributes
    ----------
        config : str
            Absolute path to text file with network configuration.
        model : str
            Absolute path to weights file of network.
        framework : str
            Name of the framework in which format network is represented.
    '''

    def __init__(self, precision='FP32'):
        self.config = '{config}'.replace('{precisionMarker}', precision)
        self.model = '{model}'.replace('{precisionMarker}', precision)
        self.framework = '{framework}'

        if not os.path.exists(self.config) or not os.path.exists(self.model):
            sys.argv = ['', '--name={name}', '--output_dir={outdir}',
                        '--precisions=' + precision]
            exec(open('{downloader}').read(), globals())


    def getIR(self, precision='FP32'):
        '''
        Creates OpenVINO Intermediate Representation (IR) network format.

        Returns
        -------
            xmlPath
                Path to generated or downloaded .xml file
            binPath
                Path to generated or downloaded .bin file
        '''
        if self.framework == 'dldt':
            return self.config, self.model

        prefix = os.path.join('{outdir}', '{name}', precision, '{name}')
        xmlPath = prefix + '.xml'
        binPath = prefix + '.bin'
        if os.path.exists(xmlPath) and os.path.exists(binPath):
            return xmlPath, binPath

        sys.argv = ['', '--name={name}', '--precisions=' + precision]
        from converter import main
        main()

        return xmlPath, binPath


    def getOCVModel(self, useIR=True):
        '''
        Creates OpenCV's cv::dnn::Model from origin network. Depends on network type,
        returns cv.dnn_DetectionModel, cv.dnn_ClassificationModel, cv.dnn_SegmentationModel
        or cv.dnn_Model if not specified.

        Preprocessing parameters are set.
        '''
        import cv2 as cv

        task_type = '{task_type}'
        mo_args = {mo_args}

        if useIR:
            model, config = self.getIR()
        else:
            model, config = self.model, self.config

        if task_type == 'detection':
            m = cv.dnn_DetectionModel(model, config)
        elif task_type == 'classification':
            m = cv.dnn_ClassificationModel(model, config)
        elif task_type == 'semantic_segmentation':
            m = cv.dnn_SegmentationModel(model, config)
        else:
            m = cv.dnn_Model(model, config)

        if useIR:
            return m

        if '--mean_values' in mo_args:
            mean = mo_args['--mean_values']
            mean = mean[mean.find('[') + 1:mean.find(']')].split(',')
            mean = [float(val) for val in mean]
            m.setInputMean(mean)

        if '--scale_values' in mo_args:
            scale = mo_args['--scale_values']
            scale = 1.0 / float(scale[scale.find('[') + 1:scale.find(']')])
            m.setInputScale(scale)

        if '--input_shape' in mo_args:
            shape = mo_args['--input_shape']
            shape = shape[shape.find('[') + 1:shape.find(']')].split(',')
            shape = [int(val) for val in shape]
            if len(shape) == 4:
                if self.framework == 'tf':  # NHWC
                    w, h = shape[2], shape[1]
                else:  # NCHW
                    w, h = shape[3], shape[2]
            m.setInputSize(w, h)

        if '--reverse_input_channels' in mo_args:
            m.setInputSwapRB(True)

        return m


    def getIENetwork(self):
        '''
        Creates openvino.inference_engine.IENetwork instance.
        Model Optimizer is lauched to create OpenVINO Intermediate Representation (IR).
        '''
        from openvino.inference_engine import IENetwork
        xmlPath, binPath = self.getIR()
        return IENetwork(xmlPath, binPath)

            """.format(className=_name, description=_topology.description,
                       license=_topology.license_url,
                       name=_topology.name,
                       outdir=str(_cache).replace('\\', '\\\\'),
                       config=str(_config_path).replace('\\', '\\\\'),
                       model=str(_model_path).replace('\\', '\\\\'),
                       framework=_topology.framework,
                       downloader=str(_downloader).replace('\\', '\\\\'),
                       precisionMarker=_precision_marker,
                       task_type=_topology.task_type,
                       mo_args=_mo_args))
