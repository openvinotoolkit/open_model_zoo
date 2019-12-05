import sys
import os
import re
import yaml
import types
import argparse

from pathlib import Path

import common

_precision_marker = '$precision'
_cache = Path(os.getenv('OPENCV_OPEN_MODEL_ZOO_CACHE_DIR', Path('.').resolve() / 'models'))


class Topology:
    def __init__(self, name, task_type, download_dir, config, model, framework,
                 mo_args, precision):
        self.name = name
        self.task_type = task_type
        self.mo_args = mo_args
        self.download_dir = download_dir
        self.config = config.replace(_precision_marker, precision)
        self.model = model.replace(_precision_marker, precision)
        self.framework = framework

        if self.config and not os.path.exists(self.config) or \
           self.model and not os.path.exists(self.model):
            sys.argv = ['', '--name=' + self.name, '--precisions=' + precision,
                        '--output_dir=' + str(_cache).replace('\\', '\\\\')]
            from downloader import main
            main()


    def get_ir(self, precision='FP32'):
        '''
        Creates OpenVINO Intermediate Representation (IR) network model.
        If there is already converted model or the model has been downloaded in
        IR - just returns the paths to the files.

        Returns
        -------
            xmlPath
                Path to generated or downloaded .xml file
            binPath
                Path to generated or downloaded .bin file
        '''
        prefix = Path(self.download_dir) / precision / self.name
        xmlPath = str(prefix) + '.xml'
        binPath = str(prefix) + '.bin'
        if os.path.exists(xmlPath) and os.path.exists(binPath):
            return xmlPath, binPath

        sys.argv = ['', '--name=' + self.name, '--precisions=' + precision,
                    '--download_dir=' + str(_cache).replace('\\', '\\\\')]
        from converter import main
        main()

        return xmlPath, binPath


    def get_ocv_model(self, use_ir=True):
        '''
        Creates OpenCV's cv.dnn_Model from origin network. Depends on network type,
        returns cv.dnn_DetectionModel, cv.dnn_ClassificationModel, cv.dnn_SegmentationModel
        or cv.dnn_Model in case of unclassified type.

        Preprocessing parameters are set.
        '''
        import cv2 as cv

        if use_ir:
            model, config = self.get_ir()
        else:
            model, config = self.model, self.config

        if self.task_type == 'detection':
            m = cv.dnn_DetectionModel(model, config)
        elif self.task_type == 'classification':
            m = cv.dnn_ClassificationModel(model, config)
        elif self.task_type == 'semantic_segmentation':
            m = cv.dnn_SegmentationModel(model, config)
        else:
            m = cv.dnn_Model(model, config)

        if use_ir:
            return m

        if '--mean_values' in self.mo_args:
            mean = self.mo_args['--mean_values']
            mean = mean[mean.find('[') + 1:mean.find(']')].split(',')
            mean = [float(val) for val in mean]
            m.setInputMean(mean)

        if '--scale_values' in self.mo_args:
            scale = self.mo_args['--scale_values']
            scale = 1.0 / float(scale[scale.find('[') + 1:scale.find(']')])
            m.setInputScale(scale)

        if '--input_shape' in self.mo_args:
            shape = self.mo_args['--input_shape']
            shape = shape[shape.find('[') + 1:shape.find(']')].split(',')
            shape = [int(val) for val in shape]
            if len(shape) == 4:
                if self.framework == 'tf':  # NHWC
                    w, h = shape[2], shape[1]
                else:  # NCHW
                    w, h = shape[3], shape[2]
            m.setInputSize(w, h)

        if '--reverse_input_channels' in self.mo_args:
            m.setInputSwapRB(True)

        return m


    def get_ie_network(self):
        '''
        Creates openvino.inference_engine.IENetwork instance.
        Model Optimizer is launched to create OpenVINO Intermediate Representation (IR).
        '''
        from openvino.inference_engine import IENetwork
        xmlPath, binPath = self.get_ir()
        return IENetwork(xmlPath, binPath)


for _topology in common.load_models(argparse.Namespace(config=None)):
    _name = _topology.name.replace('-', '_').replace('.', '_')
    _download_dir = _cache / _topology.subdirectory
    _mo_args = {}
    if _topology.mo_args:
        for _arg in _topology.mo_args:
            _tokens = _arg.split('=')
            _mo_args[_tokens[0]] = _tokens[1] if len(_tokens) == 2 else None

    # Resolve weights and text files names.
    if _topology.framework == 'dldt':
        assert(len(_topology.files) >= 2), 'Expected to have at least two files in IR format'
        _paths = []
        # DLDT topologies usually come with multiple precision configuration files.
        for i in range(2):
            # Get only basename and add precision marker
            _path = str(_topology.files[i].name)
            _path = _precision_marker + _path[_path.find('/'):]
            _paths.append(_path)

        _config_path, _model_path = _paths if _paths[0].endswith('.xml') else _paths[::-1]
    else:
        # Get paths from Model Optimizer arguments
        if '--input_model' in _mo_args:
            _model_path = _mo_args['--input_model'].replace('$dl_dir/', '')
            _config_path = ''  # If there is a text file - we will set it next

        for key in ['--input_proto', '--input_symbol']:
            if key in _mo_args:
                _config_path = _mo_args[key].replace('$dl_dir/', '')

    _model_path = _download_dir / _model_path
    if _config_path:
        _config_path = _download_dir / _config_path

    exec("""
def {funcName}(precision='FP32'):
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
    return Topology('{name}', '{task_type}', '{download_dir}', '{config}',
                    '{model}', '{framework}', {mo_args}, precision)
            """.format(funcName=_name, description=_topology.description,
                       license=_topology.license_url,
                       name=_topology.name,
                       download_dir=str(_download_dir),
                       config=str(_config_path).replace('\\', '\\\\'),
                       model=str(_model_path).replace('\\', '\\\\'),
                       framework=_topology.framework,
                       task_type=_topology.task_type,
                       mo_args=_mo_args))
