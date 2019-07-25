"""
Copyright (c) 2019 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import tensorflow as tf

from .launcher import Launcher
from ..config import BaseField, ListField, PathField, StringField, ConfigError, ConfigValidator


class TFLauncher(Launcher):
    __provider__ = 'tf'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'model': PathField(is_directory=False, description="Path to model file."),
            'device': StringField(
                choices=('cpu', 'gpu'), default='cpu', optional=True, description="Device name: cpu or gpu"),
            'inputs': BaseField(optional=True, description="Inputs."),
            'output_names': ListField(
                allow_empty=False, optional=True, value_type=StringField(), description="Output names."
            )
        })
        return parameters

    def __init__(self, config_entry, *args, **kwargs):
        super().__init__(config_entry, *args, **kwargs)
        self.default_layout = 'NHWC'

        tf_launcher_config = ConfigValidator('TF_Launcher', fields=self.parameters())
        tf_launcher_config.validate(self.config)

        self._graph = self._load_graph(str(self.get_value_from_config('model')))

        self._outputs_names = self._get_outputs_names(self._graph, self.get_value_from_config('output_names'))

        self._outputs_tensors = []
        for output in self._outputs_names:
            try:
                tensor = self._graph.get_tensor_by_name('import/{}:0'.format(output))
            except KeyError:
                raise ConfigError('model graph does not contains output {}'.format(output))
            self._outputs_tensors.append(tensor)

        self.device = '/{}:0'.format(self.get_value_from_config('device').lower())

    def predict(self, inputs, metadata, *args, **kwargs):
        """
        Args:
            inputs: dictionary where keys are input layers names and values are data for them.
            metadata: metadata of input representations
        Returns:
            raw data from network.
        """
        results = []
        for infer_input in inputs:
            with tf.device(self.device):
                with tf.Session(graph=self._graph) as session:
                    feed_dictionary = {
                        'import/{}:0'.format(input_name): input_data for input_name, input_data in infer_input.items()
                    }
                    result = session.run(self._outputs_tensors, feed_dict=feed_dictionary)
                    res = dict(zip(self._outputs_names, result))
                    results.append(res)

        return results

    @property
    def batch(self):
        return 1

    @property
    def inputs(self):
        graph_inputs = self._get_graph_inputs(self._graph)
        return {
            node_name.split('import/')[-1]:
                tuple(int(a.size) for a in node.attr['shape'].shape.dim) for node_name, node in graph_inputs.items()
        }

    def release(self):
        del self._graph

    @property
    def output_blob(self):
        return next(iter(self._outputs_names))

    def predict_async(self, *args, **kwargs):
        raise ValueError('TensorFlow Launcher does not support async mode yet')

    @staticmethod
    def _load_graph(model):
        with tf.gfile.GFile(model, 'rb') as file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def)

        return graph

    @staticmethod
    def _get_graph_inputs(graph):
        inputs_ops = {'Placeholder'}
        inputs = [x for x in graph.as_graph_def().node if not x.input and x.op in inputs_ops]

        return {node.name: node for node in inputs}

    @staticmethod
    def _get_outputs_names(graph, outputs_names=None):
        # prefer user's outputs
        if outputs_names:
            return outputs_names

        # try to find outputs
        # since there is no output attribute for node
        # we have to save all relationships parent -> child and find nodes without childes
        nodes_map = {}
        for node in graph.as_graph_def().node:
            for parent in node.input:
                nodes_map.update({parent: nodes_map.get(parent, []) + [node.name]})
        # additionally filter by operation types
        not_outputs_types = {'Const', 'Assign', 'NoOp', 'Placeholder', 'Assert'}

        names = [
            x.name.split('import/')[-1] for x in graph.as_graph_def().node
            if x.name not in nodes_map and x.op not in not_outputs_types
        ]
        if not names:
            raise ConfigError('output blobs in the graph cannot be found')

        return names
