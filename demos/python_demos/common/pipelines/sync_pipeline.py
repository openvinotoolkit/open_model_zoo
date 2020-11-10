

class SyncPipeline:
    def __init__(self, ie, model, device='CPU'):
        self.device = device
        self.model = model

        self.exec_net = ie.load_network(network=self.model.net, device_name=self.device)

    def submit_data(self, inputs):
        inputs, meta = self.model.preprocess(inputs)
        outputs = self.exec_net.infer(inputs=inputs)
        outputs = self.model.postprocess(outputs, meta)
        return outputs, meta
