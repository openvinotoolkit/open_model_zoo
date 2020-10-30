from collections import deque
import threading
from time import perf_counter


class Model:
    def __init__(self, ie, model_path, logger=None):
        self.logger = logger
        if logger:
            logger.info('Reading network from IR...')
        loading_time = perf_counter()
        self.net = ie.read_network(model_path, model_path[:-4] + '.bin')
        loading_time = (perf_counter() - loading_time)
        if logger:
            logger.info('Read in {:.3f} seconds'.format(loading_time))

    def unify_inputs(self, inputs) -> dict:
        raise NotImplementedError

    def preprocess(self, inputs):
        meta = {}
        return inputs, meta

    def postprocess(self, outputs, meta):
        return outputs
