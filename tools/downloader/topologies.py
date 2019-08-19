import sys
import os
import re
import yaml
import types

import common

if 'INTEL_OPENVINO_DIR' in os.environ:  # As a part of OpenVINO
    _loc = os.path.join(os.environ['INTEL_OPENVINO_DIR'], 'deployment_tools', 'open_model_zoo')
    _modelsDir = os.path.join(_loc, 'models')
    _downloader = os.path.join(_loc, 'tools', 'downloader', 'downloader.py')
else:  # Standalone
    _loc = os.path.dirname(__file__)
    _modelsDir = os.path.join(_loc, '..', '..', 'models')
    _downloader = os.path.join(_loc, 'downloader.py')

_precisionMarker = '$precision'
_cache = os.path.abspath(os.getenv('OPENCV_OPEN_MODEL_ZOO_CACHE_DIR', 'models'))

# Parse topologies.
for _moduleName in os.listdir(_modelsDir):
    _modelsSubDir = os.path.join(_modelsDir, _moduleName)
    if not os.path.isdir(_modelsSubDir):
        continue

    _module = types.ModuleType(_moduleName)
    sys.modules['topologies.' + _moduleName] = _module
    globals()[_moduleName] = _module
    _module.__dict__['sys'] = sys
    _module.__dict__['os'] = os

    # List of models which have versions (mostly DLDT models). For such kind of models
    # with the highest version we will add short names. In example, for
    # "face_detection_retail_0004" and "face_detection_retail_0005" method with name
    # "face_detection_retail" returns result of "face_detection_retail_0005".
    _versionedNames = {}

    for _topologyName in os.listdir(_modelsSubDir):
        _config = os.path.join(_modelsSubDir, _topologyName, 'model.yml')
        if not os.path.exists(_config):
            continue

        with open(_config, 'rt') as _f:
            _topology = yaml.safe_load(_f)
            _name = _topologyName.replace('-', '_').replace('.', '_')
            _description = _topology['description']
            _license = _topology['license']
            _files = _topology['files']
            _framework = _topology['framework']
            _taskType = _topology['task_type']
            _configPath = ''
            _modelPath = ''
            _downloadDir = os.path.join(_cache, _moduleName, _topologyName)
            _moArgs = {}
            if 'model_optimizer_args' in _topology:
                for _arg in _topology['model_optimizer_args']:
                    _tokens = _arg.split('=')
                    _moArgs[_tokens[0]] = _tokens[1] if len(_tokens) == 2 else None


            assert(len(_files) > 0)
            if len(_files) > 2:
                assert(_framework == 'dldt'), ('Unexpected framework type: ' + _framework)
                # Get only basename and add precision marker
                _configPath = _files[0]['name']
                _modelPath = _files[1]['name']

                _pos = _configPath.find('/')
                if _configPath[:_pos] in common.KNOWN_PRECISIONS:
                    _configPath = _precisionMarker + _configPath[_pos:]

                _pos = _modelPath.find('/')
                if _modelPath[:_pos] in common.KNOWN_PRECISIONS:
                    _modelPath = _precisionMarker + _modelPath[_pos:]

            else:
                _configPath = _files[0]['name']
                _modelPath = _files[-1]['name']

            # To manage origin files location from archives
            if '--input_model' in _moArgs:
                _modelPath = _moArgs['--input_model'].replace('$dl_dir/', '')
            if '--input_proto' in _moArgs:
                _configPath = _moArgs['--input_proto'].replace('$dl_dir/', '')

            if _configPath:
                _configPath = os.path.realpath(os.path.join(_downloadDir, _configPath))
            if _modelPath:
                _modelPath = os.path.realpath(os.path.join(_downloadDir, _modelPath))


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

        prefix = os.path.join('{outdir}', '{module}', '{name}', precision, '{name}')
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

            """.format(className=_name, description=_description, license=_license,
                       name=_topologyName, outdir=_cache,
                       config=_configPath, model=_modelPath,
                       framework=_framework, downloader=_downloader, module=_moduleName,
                       precisionMarker=_precisionMarker, task_type=_taskType,
                       mo_args=_moArgs),
            _module.__dict__)

            # Extract version and add an alias.
            _matches = re.search('(.+)_(\d{4})$', _name)
            if _matches:
                _shortName = _matches.group(1)
                _version = _matches.group(2)
                if not _shortName in _versionedNames or int(_version) > int(_versionedNames[_shortName]):
                    _versionedNames[_shortName] = _version
                    exec("""
{alias} = {className}
                    """.format(alias=_shortName, className=_name),
                    _module.__dict__)
