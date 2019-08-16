import sys
import os
import yaml

# TODO: Decide where open_model_zoo is located in OpenVINO relatively to this script
_modelsDir = os.path.join('..', '..', 'models', 'public')

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

# TODO: topologies.public and topologies.intel submodules
# Parse topologies.
for _topologyName in os.listdir(_modelsDir):
    _config = os.path.join(_modelsDir, _topologyName, 'model.yml')
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
        _downloadDir = os.path.join(_cache, 'public', _topologyName)

        assert(len(_files) > 0)
        if len(_files) > 2:
            assert(_framework == 'dldt'), ('Unexpected framework type: ' + _framework)
            pass
        else:
            _configPath = os.path.join(_downloadDir, _files[0]['name'])
            _modelPath = os.path.join(_downloadDir, _files[-1]['name'])

        # To manage origin files location from archives
        if 'model_optimizer_args' in _topology:
            for arg in _topology['model_optimizer_args']:
                tokens = arg.split('=')
                if len(tokens) == 2:
                    if tokens[0] == '--input_model':
                        _modelPath = os.path.realpath(tokens[1].replace('$dl_dir', _downloadDir))
                    elif tokens[0] == '--input_proto':
                        _configPath = os.path.realpath(tokens[1].replace('$dl_dir', _downloadDir))


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

    def __init__(self):
        self.config = '{config}'
        self.model = '{model}'
        self.framework = '{framework}'

        sys.argv = ['', '--name={name}', '--output_dir={outdir}', '--cache_dir={cache}']
        exec(open('downloader.py').read(), globals())


    def getIR(self, precision='{defaultPrecision}'):
        if self.framework == 'dldt':
            return self.config, self.model

        prefix = os.path.join('{outdir}', 'public', '{name}', precision, '{name}')
        xmlPath = prefix + '.xml'
        binPath = prefix + '.bin'
        if os.path.exists(xmlPath) and os.path.exists(binPath):
            return xmlPath, binPath

        sys.argv = ['', '--name={name}', '--precision=' + precision]
        from converter import main
        main()

        return xmlPath, binPath

        """.format(className=_name, description=_description, license=_license,
                   name=_topologyName, outdir=_cache, cache=_cache,
                   config=_configPath, model=_modelPath, defaultPrecision='FP32',
                   framework=_framework))
