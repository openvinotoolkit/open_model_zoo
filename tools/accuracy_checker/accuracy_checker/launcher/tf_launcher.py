"""
Copyright (c) 2018-2020 Intel Corporation

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

import re
from pathlib import Path
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from .launcher import Launcher, LauncherConfigValidator
from ..config import BaseField, ListField, PathField, StringField, ConfigError
from ..utils import contains_any, contains_all


class TFLauncher(Launcher):
    __provider__ = 'tf'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'model': PathField(
                is_directory=False, description="Path to model file (frozen graph of checkpoint meta).", optional=True
            ),
            'saved_model_dir': PathField(is_directory=True, optional=True, description='Path to saved model directory'),
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
        self._delayed_model_loading = kwargs.get('delayed_model_loading', False)

        tf_launcher_config = LauncherConfigValidator(
            'TF_Launcher', fields=self.parameters(), delayed_model_loading=self._delayed_model_loading
        )
        tf_launcher_config.validate(self.config)

        if not self._delayed_model_loading:
            if not contains_any(self.config, ['model', 'saved_model_dir']):
                raise ConfigError('model or saved model directory should be provided')

            if contains_all(self.config, ['model', 'saved_model']):
                raise ConfigError('only one option: model or saved_model_dir should be provided')

            self._config_outputs = self.get_value_from_config('output_names')
            if 'model' in self.config:
                self._graph = self._load_graph(str(self.get_value_from_config('model')))
            else:
                self._graph = self._load_graph(str(self.get_value_from_config('saved_model_dir')), True)

            self._outputs_names = self._get_outputs_names(self._graph, self._config_outputs)

            self._outputs_tensors = []
            self.node_pattern = 'import/{}:0'
            for output in self._outputs_names:
                try:
                    tensor = self._graph.get_tensor_by_name('import/{}:0'.format(output))
                except KeyError:
                    try:
                        tensor = self._graph.get_tensor_by_name('{}:0'.format(output))
                        self.node_pattern = '{}:0'
                    except KeyError:
                        raise ConfigError('model graph does not contains output {}'.format(output))
                self._outputs_tensors.append(tensor)

        self.device = '/{}:0'.format(self.get_value_from_config('device').lower())

    def predict(self, inputs, metadata=None, **kwargs):
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
                        self.node_pattern.format(input_name): input_data
                        for input_name, input_data in infer_input.items()
                    }
                    result = session.run(self._outputs_tensors, feed_dict=feed_dictionary)
                    res = dict(zip(self._outputs_names, result))
                    results.append(res)
            if metadata is not None:
                for meta_ in metadata:
                    meta_['input_shape'] = self.inputs_info_for_meta()

        return results

    @property
    def batch(self):
        return 1

    @property
    def inputs(self):
        graph_inputs = self._get_graph_inputs(self._graph, self.get_value_from_config('inputs'))
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

    def _load_graph(self, model, saved_model=False):
        if saved_model:
            return self._load_saved_model(model)

        if 'meta' in Path(model).suffix:
            return self._load_graph_using_meta(model)

        return self._load_frozen_graph(model)

    def _load_graph_using_meta(self, model):
        tf.reset_default_graph()
        graph = tf.Graph()
        graph_def = tf.MetaGraphDef()

        with open(model, "rb") as model_file:
            graph_def.ParseFromString(model_file.read())

        with tf.Session() as sess:
            restorer = tf.train.import_meta_graph(graph_def)
            restorer.restore(sess, re.sub(r'\.meta$', '', model))
            graph_def = tf.graph_util.convert_variables_to_constants(
                sess, graph_def.graph_def, self._config_outputs
            )

        with graph.as_default():
            tf.import_graph_def(graph_def, name='')
        return graph

    @staticmethod
    def _load_frozen_graph(model):
        with tf.gfile.GFile(model, 'rb') as file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def)

        return graph

    @staticmethod
    def _load_saved_model(model_dir):
        graph = tf.Graph()

        with graph.as_default():
            with tf.Session() as sess:
                tf.saved_model.loader.load(sess, [tag_constants.SERVING], model_dir)

        return graph

    def _get_graph_inputs(self, graph, config_inputs=None):
        inputs_ops = {'Placeholder'}
        inputs = [x for x in graph.as_graph_def().node if not x.input and x.op in inputs_ops]
        if config_inputs:
            node_pattern_without_op = self.node_pattern.split(':')[0]
            config_inputs_names = [node_pattern_without_op.format(layer['name']) for layer in config_inputs]
            config_inputs = [x for x in graph.as_graph_def().node if x.name in config_inputs_names]
            inputs.extend(config_inputs)

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
