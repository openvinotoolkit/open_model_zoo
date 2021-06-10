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
from pathlib import Path
import numpy as np
from .launcher import Launcher
from ..config import PathField, StringField, ConfigError
from ..logging import print_info

DEVICE_REGEX = r'(?P<device>cpu$|gpu)'


class PaddlePaddleLauncher(Launcher):
    __provider__ = 'paddle_paddle'

    def __init__(self, config_entry: dict, *args, **kwargs):
        super().__init__(config_entry, *args, **kwargs)
        self._delayed_model_loading = kwargs.get('delayed_model_loading', False)
        try:
            from paddle.fluid.core import PaddleTensor, AnalysisConfig, create_paddle_predictor # pylint: disable=C0415
        except ImportError as import_error:
            raise ValueError(
                "PaddlePaddle isn't installed. Please, install it before using. \n{}".format(import_error.msg)
            )
        self._paddle_tensor = PaddleTensor
        self._analysis_config = AnalysisConfig
        self._create_paddle_predictor = create_paddle_predictor
        self.validate_config(config_entry, delayed_model_loading=self._delayed_model_loading)
        if not self._delayed_model_loading:
            self.load_model()

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'model': PathField(description="Path to model.", file_or_directory=True),
            'params': PathField(description='Path to model params', optional=True, file_or_directory=True),
            'device': StringField(regex=DEVICE_REGEX, description="Device name.", optional=True, default='CPU')
        })

        return parameters

    def load_model(self):
        model, params = self.automatic_model_search()
        config = self._analysis_config(str(model), str(params))
        config.disable_glog_info()
        device = self.get_value_from_config('device')
        if device.upper() == 'CPU':
            config.disable_gpu()
        self._predictor = self._create_paddle_predictor(config)

    @property
    def inputs(self):
        return self._predictor.get_input_tensor_shape()

    @property
    def output_blob(self):
        return next(iter(self._predictor.get_output_names()))

    @property
    def batch(self):
        return 1

    def automatic_model_search(self):
        model = Path(self.get_value_from_config('model'))
        if model.is_dir():
            model_list = list(model.glob('{}.pdmodel'.format(self._model_name)))
            if not model_list:
                model_list = list(model.glob('*.pdmodel'))
                if not model_list:
                    raise ConfigError('Model not found')
            model = model_list[0]

        params = self.get_value_from_config('params')
        if params is None or Path(params).is_dir():
            params_dir = model.parent if params is None else Path(params)
            params_list = list(params_dir.glob('{}.pdiparams'.format(self._model_name)))
            if not params_list:
                params_list = list(params_dir.glob('*.pdiparams'))
                if not params_list:
                    raise ConfigError('Params not found')
            params = params_list[0]
        accepted_suffixes = ['.pdmodel']
        if model.suffix not in accepted_suffixes:
            raise ConfigError('Models with following suffixes are allowed: {}'.format(accepted_suffixes))
        print_info('Found model {}'.format(model))
        params = Path(params)
        accepted_params_suffixes = ['.pdiparams']
        if params.suffix not in accepted_params_suffixes:
            raise ConfigError('Params with following suffixes are allowed: {}'.format(accepted_params_suffixes))
        print_info('Found weights {}'.format(params))

        return model, params

    def predict(self, inputs, metadata=None, **kwargs):
        results = []
        for infer_input in inputs:
            input_list = [infer_input[name] for name in self._predictor.get_input_names()]
            prediction_list = self._predictor.run(input_list)
            res_outs = {}
            for name, prediction in zip(self._predictor.get_output_names(), prediction_list):
                res_outs[name] = prediction.as_ndarray()
            results.append(res_outs)
            if metadata is not None:
                for meta_ in metadata:
                    meta_['input_shape'] = self.inputs_info_for_meta()

        return results

    def fit_to_input(self, data, layer_name, layout, precision):
        layer_shape = self.inputs[layer_name]
        input_precision = np.float32 if not precision else precision
        if len(np.shape(data)) == 4:
            if layout is not None:
                data = np.transpose(data, layout)
            data = data.astype(input_precision)
            if len(layer_shape) == 3:
                if np.shape(data)[0] != 1:
                    raise ValueError('Only for batch size 1 first dimension can be omitted')
                return self._paddle_tensor(data[0].astype(input_precision))
            return self._paddle_tensor(data.astype(input_precision))
        if layout is not None and len(np.shape(data)) == 5 and len(layout) == 5:
            return self._paddle_tensor(np.transpose(data, layout).astype(input_precision))
        if len(np.shape(data))-1 == len(layer_shape):
            return self._paddle_tensor(np.array(data[0]).astype(input_precision))
        return self._paddle_tensor(np.array(data).astype(input_precision))

    def predict_async(self, *args, **kwargs):
        raise ValueError('PaddlePaddle Launcher does not support async mode yet')

    def release(self):
        if hasattr(self, '_predictor'):
            del self._predictor
