from time import perf_counter


class Model:
    def __init__(self, ie, model_path, logger=None, batch_size=1):
        self.logger = logger
        if self.logger:
            self.logger.info('Reading network from IR...')
        loading_time = perf_counter()
        self.net = ie.read_network(model_path, model_path[:-4] + '.bin')
        loading_time = (perf_counter() - loading_time)
        if self.logger:
            self.logger.info('Read in {:.3f} seconds'.format(loading_time))
        if batch_size:
            self.set_batch_size(batch_size)

    def preprocess(self, inputs):
        meta = {}
        return inputs, meta

    def postprocess(self, outputs, meta):
        return outputs

    def set_batch_size(self, batch):
        shapes = {}
        for input_layer in self.net.input_info:
            new_shape = [batch] + self.net.input_info[input_layer].input_data.shape[1:]
            shapes.update({input_layer: new_shape})
        self.net.reshape(shapes)
