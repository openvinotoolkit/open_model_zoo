import sys
import os
import yaml

# TODO: Decide where open_model_zoo is located in OpenVINO relatively to this script
_modelsDir = os.path.join('..', '..', 'models', 'public')

# Determine cache directory
_cache = None
_cacheEnv = 'OPENCV_OPEN_MODEL_ZOO_CACHE_DIR'
_cache = os.getenv('OPENCV_OPEN_MODEL_ZOO_CACHE_DIR', None)
if not _cache:
    try:
        import cv2 as cv
        _cache = cv.utils.fs.getCacheDirectory('open_model_zoo', _cacheEnv)
    except:
        pass
assert(_cache), 'Cache must be specified by OPENCV_OPEN_MODEL_ZOO_CACHE_DIR environment variable'

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

        exec("""
class %s:
    '''
    %s

    License: %s
    '''

    def __init__(self):
        sys.argv = ['', '--name=%s', '--output_dir=%s', '--cache_dir=%s']
        exec(open('downloader.py').read(), globals())
        """ % (_name, _description, _license, _topologyName, _cache, _cache))
