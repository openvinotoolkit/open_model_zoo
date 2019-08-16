import sys
import os
import yaml
import types

import common

_loc = os.path.dirname(__file__)

# TODO: Decide where open_model_zoo is located in OpenVINO relatively to this script
_modelsDir = os.path.join(_loc, '..', '..', 'models')
_downloader = os.path.join(_loc, 'downloader.py')
_precisionMarker = '$precision'

# Determine cache directory
_cache = None
_cacheEnv = 'OPENCV_OPEN_MODEL_ZOO_CACHE_DIR'
_cache = os.getenv(_cacheEnv, None)
if not _cache:
    try:
        import cv2 as cv
        _cache = cv.utils.fs.getCacheDirectory('open_model_zoo', _cacheEnv)
    except:
        pass
assert(_cache), 'Cache must be specified by ' + _cacheEnv + ' environment variable'
_cache = os.path.abspath(_cache)


# Parse topologies.
for _moduleName in os.listdir(_modelsDir):
    _module = types.ModuleType(_moduleName)
    sys.modules['topologies.' + _moduleName] = _module
    globals()[_moduleName] = _module
    _module.__dict__['sys'] = sys
    _module.__dict__['os'] = os

    _modelsSubDir = os.path.join(_modelsDir, _moduleName)
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
            _configPath = ''
            _modelPath = ''
            _downloadDir = os.path.join(_cache, _moduleName, _topologyName)

            assert(len(_files) > 0)
            if len(_files) > 2:
                assert(_framework == 'dldt'), ('Unexpected framework type: ' + _framework)
                # Get only basename and add precision marker
                _configPath = _files[0]['name']
                _modelPath = _files[1]['name']

                pos = _configPath.find('/')
                if _configPath[:pos] in common.KNOWN_PRECISIONS:
                    _configPath = _precisionMarker + _configPath[pos:]

                pos = _modelPath.find('/')
                if _modelPath[:pos] in common.KNOWN_PRECISIONS:
                    _modelPath = _precisionMarker + _modelPath[pos:]

            else:
                _configPath = os.path.join(_downloadDir, _files[0]['name'])
                _modelPath = os.path.join(_downloadDir, _files[-1]['name'])

            # To manage origin files location from archives
            if 'model_optimizer_args' in _topology:
                for _arg in _topology['model_optimizer_args']:
                    _tokens = _arg.split('=')
                    if len(_tokens) == 2:
                        if _tokens[0] == '--input_model':
                            _modelPath = _tokens[1].replace('$dl_dir', _downloadDir)
                        elif _tokens[0] == '--input_proto':
                            _configPath = _tokens[1].replace('$dl_dir', _downloadDir)

            _modelPath = os.path.realpath(_modelPath)
            _configPath = os.path.realpath(_configPath)


            exec("""
class {className}:
    '''
    {description}

    License: {license}

    Attributes:
        config (str): Absolute path to text file with network configuration.
        model (str): Absolute path to weights file of network.
        framework (str): Name of the framework in which format network is represented.
    '''

    def __init__(self, precision='{defaultPrecision}'):
        self.config = '{config}'.replace('{precisionMarker}', precision)
        self.model = '{model}'.replace('{precisionMarker}', precision)
        self.framework = '{framework}'

        sys.argv = ['', '--name={name}', '--output_dir={outdir}',
                    '--cache_dir={cache}', '--precisions=' + precision]
        exec(open('{downloader}').read(), globals())


    def getIR(self, precision='{defaultPrecision}'):
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

            """.format(className=_name, description=_description, license=_license,
                       name=_topologyName, outdir=_cache, cache=_cache,
                       config=_configPath, model=_modelPath, defaultPrecision='FP32',
                       framework=_framework, downloader=_downloader, module=_moduleName,
                       precisionMarker=_precisionMarker),
            _module.__dict__)
