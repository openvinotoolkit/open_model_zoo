import numpy as np


class Statistic:
    def __init__(self, functor, batch_size):
        self.iter_counter = 0
        self.state = np.array([])
        self.processor = functor
        self.batch_size = batch_size

    def update(self, activation):
        an = self.processor(activation)
        if self.iter_counter == 0:
            self.state = an
        else:
            self.state = (self.state * self.iter_counter + an) / (self.iter_counter + 1)
        self.iter_counter += 1

    def update_on_batch(self, batch_a):
        an_shape = np.shape(batch_a)
        if an_shape[0] != self.batch_size:
            self.update(batch_a)
            return
        for activation in batch_a:
            self.update(activation)


class StatisticsCollector:
    def __init__(self, functors_mapping, batch=1):
        self.statistics = {}
        for layer_name, functors in functors_mapping.items():
            self.statistics[layer_name] = [Statistic(functor, batch) for functor in functors]

    def process_batch(self, outputs):
        output_dict = outputs[0]
        for layer_name, output in output_dict.items():
            if layer_name not in self.statistics:
                continue

            for statistic in self.statistics[layer_name]:
                statistic.update_on_batch(output)

    def get_statistics(self):
        per_layer_statistics = {}
        for layer_name, layer_statistics in self.statistics.items():
            per_layer_statistics[layer_name] = [statistic.state for statistic in layer_statistics]

        return per_layer_statistics
