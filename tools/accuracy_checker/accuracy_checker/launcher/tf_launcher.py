"""
Copyright (c) 2018-2021 Intel Corporation

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
import numpy as np

from .launcher import Launcher
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
        try:
            import tensorflow # pylint: disable=C0415
            from tensorflow.python.saved_model import tag_constants # pylint: disable=C0415
            if tensorflow.__version__ >= '2.0.0':
                self.tf = tensorflow.compat.v1
                self.tf_gfile = tensorflow.io.gfile
            else:
                self.tf = tensorflow
                self.tf_gfile = tensorflow.gfile
            self.tag_constants = tag_constants
        except ImportError as import_error:
            raise ValueError(
                "TensorFlow isn't installed. Please, install it before using. \n{}".format(import_error.msg)
            )
        self.default_layout = 'NHWC'
        self._delayed_model_loading = kwargs.get('delayed_model_loading', False)
        self.validate_config(config_entry, delayed_model_loading=self._delayed_model_loading)
        self._graph = None
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
        self._output_layouts = {}
        self._lstm_inputs = None
        if '_list_lstm_inputs' in self.config:
            self._configure_lstm_inputs()

    @staticmethod
    def _data_to_blob(layer_shape, data, layout): # pylint:disable=R0911
        data_shape = np.shape(data)
        if len(layer_shape) == 4:
            if len(data_shape) == 5:
                data = data[0]
            if len(data_shape) < 4:
                if len(np.squeeze(np.zeros(layer_shape))) == len(np.squeeze(np.zeros(data_shape))):
                    return np.resize(data, layer_shape)
            return np.transpose(data, layout) if layout is not None else data
        if len(layer_shape) == 2:
            if len(data_shape) == 1:
                return np.transpose([data])
            if len(data_shape) > 2:
                if all(dim == 1 for dim in layer_shape) and all(dim == 1 for dim in data_shape):
                    return np.resize(data, layer_shape)
                if len(np.squeeze(np.zeros(layer_shape))) == len(np.squeeze(np.zeros(data_shape))):
                    return np.resize(data, layer_shape)
        if len(layer_shape) == 3 and len(data_shape) == 4:
            return np.transpose(data, layout)[0] if layout is not None else data[0]
        if layout is not None and len(layer_shape) == len(layout):
            return np.transpose(data, layout)
        if (
                len(layer_shape) == 1 and len(data_shape) > 1 and
                len(np.squeeze(np.zeros(layer_shape))) == len(np.squeeze(np.zeros(data_shape)))
        ):
            return np.resize(data, layer_shape)
        return np.array(data)

    def _set_precision(self):
        has_info = hasattr(self.network if self.network is not None else self.exec_network, 'input_info')
        config_inputs = self.config.get('inputs', [])
        for input_config in config_inputs:
            if 'precision' in input_config:
                if self.network:
                    if not has_info:
                        self.network.inputs[input_config['name']].precision = input_config['precision']
                    else:
                        self.network.input_info[input_config['name']].precision = input_config['precision']

    def _set_input_shape(self):
        if not self.network:
            return
        config_inputs = self.config.get('inputs', [])
        input_shapes = {}
        for input_config in config_inputs:
            if 'shape' in input_config:
                input_shapes[input_config['name']] = input_config['shape']
        if not input_shapes:
            return
        orig_input_shapes = {input_name: input_info.shape for input_name, input_info in self.inputs.items()}
        orig_input_shapes.update(input_shapes)
        self._reshape_input(orig_input_shapes)

    def _configure_lstm_inputs(self):
        lstm_mapping = {}
        config_inputs = self.config.get('inputs', [])
        for input_config in config_inputs:
            if input_config['type'] == 'LSTM_INPUT':
                lstm_mapping[input_config['name']] = input_config['value']
        self._lstm_inputs = lstm_mapping

    def _fill_lstm_inputs(self, infer_outputs=None):
        feed_dict = {}
        for lstm_var, output_layer in self._lstm_inputs.items():
            layer_shape = self.inputs[lstm_var]
            input_data = infer_outputs[output_layer].reshape(layer_shape) if infer_outputs else np.zeros(layer_shape)
            feed_dict[lstm_var] = input_data
        return feed_dict

    def _predict_sequential(self, inputs, metadata=None, **kwargs):
        lstm_inputs_feed = self._fill_lstm_inputs()
        results = []
        for feed_dict in inputs:
            feed_dict.update(lstm_inputs_feed)

            with self.tf.device(self.device):
                with self.tf.Session(graph=self._graph) as session:
                    feed_dictionary = {
                        self.node_pattern.format(input_name): input_data
                        for input_name, input_data in feed_dict.items()
                    }
                    result = session.run(self._outputs_tensors, feed_dict=feed_dictionary)
                    res = dict(zip(self._outputs_names, result))
                    results.append(res)


            lstm_inputs_feed = self._fill_lstm_inputs(res)

        if metadata is not None:
            for meta_ in metadata:
                meta_['input_shape'] = self.inputs_info_for_meta()
                if self._output_layouts:
                    meta_['output_layout'] = self._output_layouts

        return results

    def predict(self, inputs, metadata=None, **kwargs):
        """
        Args:
            inputs: dictionary where keys are input layers names and values are data for them.
            metadata: metadata of input representations
        Returns:
            raw data from network.
        """
        if self._lstm_inputs:
            return self._predict_sequential(inputs, metadata)

        results = []
        for infer_input in inputs:
            with self.tf.device(self.device):
                with self.tf.Session(graph=self._graph) as session:
                    feed_dictionary = {
                        self.node_pattern.format(input_name): input_data
                        for input_name, input_data in infer_input.items()
                    }
                    result = session.run(self._outputs_tensors, feed_dict=feed_dictionary)
                    res = dict(zip(self._outputs_names, result))
                    results.append(res)
            if metadata is not None:
                for meta_ in metadata:
                    meta_['input_shape'] = meta_.get('input_shape', {})
                    meta_['input_shape'].update({name: data.shape for name, data in infer_input.items()})

        return results

    def inputs_info_for_meta(self, feed_dict=None):
        if feed_dict is None:
            return super().inputs_info_for_meta()
        return {input_name: input_data.shape for input_name, input_data in feed_dict.items()}

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
        if self._graph is not None:
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
        self.tf.reset_default_graph()
        graph = self.tf.Graph()
        graph_def = self.tf.MetaGraphDef()

        with open(model, "rb") as model_file:
            graph_def.ParseFromString(model_file.read())

        with self.tf.Session() as sess:
            restorer = self.tf.train.import_meta_graph(graph_def)
            restorer.restore(sess, re.sub(r'\.meta$', '', model))
            graph_def = self.tf.graph_util.convert_variables_to_constants(
                sess, graph_def.graph_def, self._config_outputs
            )

        with graph.as_default():
            self.tf.import_graph_def(graph_def, name='')
        return graph

    def _load_frozen_graph(self, model):
        with self.tf_gfile.GFile(model, 'rb') as file:
            graph_def = self.tf.GraphDef()
            graph_def.ParseFromString(file.read())

        with self.tf.Graph().as_default() as graph:
            self.tf.import_graph_def(graph_def)

        return graph

    def _load_saved_model(self, model_dir):
        graph = self.tf.Graph()

        with graph.as_default():
            with self.tf.Session() as sess:
                self.tf.saved_model.loader.load(sess, [self.tag_constants.SERVING], model_dir)

        return graph

    def fit_to_input(self, data, layer_name, layout, precision):
        layer_shape = self.inputs[layer_name]
        if (
                len(layer_shape) > len(np.shape(data)) and
                len(np.squeeze(np.zeros(layer_shape))) == len(np.squeeze(np.zeros(np.shape(data))))
        ):
            if -1 not in layer_shape:
                data = np.resize(data, layer_shape)

        if len(np.shape(data)) == len(layout):
            data = np.transpose(data, layout)
        else:
            data = np.array(data)
        return data.astype(precision) if precision else data

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

    def create_inference_session(self, model, saved_model_dir=False, inputs=None, outputs=None):
        if saved_model_dir:
            _graph = self._load_graph(str(model), True)
        else:
            _graph = self._load_graph(str(model))

        _outputs_names = self._get_outputs_names(_graph, outputs)
        _outputs_tensors = []
        _node_pattern = 'import/{}:0'
        for output in _outputs_names:
            try:
                tensor = _graph.get_tensor_by_name('import/{}:0'.format(output))
            except KeyError:
                try:
                    tensor = _graph.get_tensor_by_name('{}:0'.format(output))
                    _node_pattern = '{}:0'
                except KeyError:
                    raise ConfigError('model graph does not contains output {}'.format(output))
            _outputs_tensors.append(tensor)

        graph_inputs = self._get_graph_inputs(_graph, inputs)
        _inputs = {
            node_name.split('import/')[-1]:
                tuple(int(a.size) for a in node.attr['shape'].shape.dim) for node_name, node in graph_inputs.items()
        }

        return TFSessionWrapper(self.tf, self.device, _graph, _outputs_names, _outputs_tensors, _inputs, _node_pattern)

class TFSessionWrapper:
    def __init__(self, tf, device, graph, outputs_names, outputs_tensors, inputs, node_pattern):
        self._tf = tf
        self._device = device
        self._graph = graph
        self._outputs_names = outputs_names
        self._outputs_tensors = outputs_tensors
        self._inputs = inputs
        self._node_pattern = node_pattern

        self.default_layout = 'NHWC'

        with self._tf.device(self._device):
            self._session = self._tf.Session(graph=self._graph)

    def predict(self, inputs, metadata=None, **kwargs):
        results = []
        for infer_input in inputs:
            feed_dictionary = {
                self._node_pattern.format(input_name): input_data
                for input_name, input_data in infer_input.items()
            }
            result = self._session.run(self._outputs_tensors, feed_dict=feed_dictionary)
            res = dict(zip(self._outputs_names, result))
            results.append(res)

            if metadata is not None:
                for meta_ in metadata:
                    meta_['input_shape'] = meta_.get('input_shape', {})
                    meta_['input_shape'].update({name: data.shape for name, data in infer_input.items()})

        return results
