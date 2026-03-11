"""
Copyright (c) 2018-2024 Intel Corporation

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

from contextlib import contextmanager
import sys
import importlib
import urllib
import re
from collections import OrderedDict
import numpy as np
from ..config import PathField, StringField, DictField, NumberField, ListField, BoolField
from ..utils import UnsupportedPackage
from .launcher import Launcher
try:
    import transformers
except ImportError as transformers_error:
    transformers = UnsupportedPackage('transformers', transformers_error.msg)

CLASS_REGEX = r'(?:\w+)'
MODULE_REGEX = r'(?:\w+)(?:(?:.\w+)*)'
DEVICE_REGEX = r'(?P<device>cpu$|cuda)?'
CHECKPOINT_URL_REGEX = r'^https?://.*\.pth(\?.*)?(#.*)?$'
SCALAR_INPUTS = ('input_ids', 'input_mask', 'segment_ids', 'attention_mask', 'token_type_ids')

class PyTorchLauncher(Launcher):
    __provider__ = 'pytorch'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'module': StringField(regex=MODULE_REGEX, description='Network module for loading'),
            'checkpoint': PathField(
                check_exists=True, is_directory=False, optional=True, description='pre-trained model checkpoint'
            ),
            'checkpoint_url': StringField(
                optional=True, regex=CHECKPOINT_URL_REGEX, description='Url link to pre-trained model checkpoint.'
            ),
            'state_key': StringField(optional=True, regex=r'\w+', description='pre-trained model checkpoint state key'),
            'python_path': PathField(
                check_exists=True, is_directory=True, optional=True,
                description='appendix for PYTHONPATH for making network module visible in current python environment'
            ),
            'module_args': ListField(optional=True, description='positional arguments for network module'),
            'module_kwargs': DictField(
                key_type=str, validate_values=False, optional=True, default={},
                description='keyword arguments for network module'
            ),
            'init_method': StringField(
                optional=True, regex=r'\w+', description='Method name to be called for module initialization.'
            ),
            'device': StringField(default='cpu', regex=DEVICE_REGEX),
            'batch': NumberField(value_type=int, min_value=1, optional=True, description="Batch size.", default=1),
            'output_names': ListField(
                optional=True, value_type=str, description='output tensor names'
            ),
            'use_torch_compile': BoolField(
                optional=True, default=False, description='Use torch.compile to optimize the module code'),
            'torch_compile_kwargs': DictField(
                key_type=str, validate_values=False, optional=True, default={},
                description="dictionary of keyword arguments passed to torch.compile"
            ),
            'transformers_class': StringField(
                optional=True, regex=CLASS_REGEX, description='Transformers class name to load pre-trained module.'
            ),
            'ultralytics_raw_output': BoolField(
                optional=True,
                default=False,
                description='Run underlying Ultralytics torch module directly and pass raw head output to adapter.'
            ),
            'ultralytics_raw_scores_sigmoid': BoolField(
                optional=True,
                default=True,
                description='Apply sigmoid to class scores when building raw Ultralytics outputs for detection adapter.'
            ),
            'ultralytics_raw_branch': StringField(
                optional=True,
                default='one2one',
                regex=r'one2one|one2many|auto',
                description='Select end2end branch for Ultralytics raw output conversion.'
            )
        })
        return parameters

    def __init__(self, config_entry: dict, *args, **kwargs):
        super().__init__(config_entry, *args, **kwargs)
        try:
            # PyTorch import affects performance of common pipeline
            # it is the reason, why it is imported only when it used
            import torch # pylint: disable=C0415
        except ImportError as import_error:
            raise ValueError("PyTorch isn't installed. Please, install it before using. \n{}".format(
                import_error.msg)) from import_error
        self._torch = torch
        self.validate_config(config_entry)
        self.use_torch_compile = config_entry.get('use_torch_compile', False)
        self.compile_kwargs = config_entry.get('torch_compile_kwargs', {})
        self.tranformers_class = config_entry.get('transformers_class', None)
        self.ultralytics_raw_output = config_entry.get('ultralytics_raw_output', False)
        self.ultralytics_raw_scores_sigmoid = config_entry.get('ultralytics_raw_scores_sigmoid', True)
        self.ultralytics_raw_branch = config_entry.get('ultralytics_raw_branch', 'one2one')
        backend = self.compile_kwargs.get('backend', None)
        if self.use_torch_compile and backend == 'openvino':
            try:
                importlib.import_module('openvino.torch')  # pylint: disable=C0415, W0611
            except ImportError as import_error:
                raise ValueError("torch.compile is supported from OpenVINO 2023.1\n{}".format(
                    import_error.msg)) from import_error
        module_args = config_entry.get("module_args", ())
        module_kwargs = config_entry.get("module_kwargs", {})
        self.device = self.get_value_from_config('device')
        self.cuda = 'cuda' in self.device

        checkpoint = config_entry.get('checkpoint')
        if checkpoint is None:
            checkpoint = config_entry.get('checkpoint_url')

        python_path = config_entry.get("python_path")

        if self.tranformers_class:
            self.module = self.load_tranformers_module(
                config_entry['module'], python_path
            )
        else:

            self.module = self.load_module(
                config_entry['module'],
                module_args,
                module_kwargs,
                checkpoint,
                config_entry.get('state_key'),
                python_path,
                config_entry.get("init_method")
            )

        self._batch = self.get_value_from_config('batch')
        # torch modules does not have input information
        self._generate_inputs()
        self.output_names = self.get_value_from_config('output_names') or ['output']

    def _generate_inputs(self):
        config_inputs = self.config.get('inputs')
        if not config_inputs:
            self._inputs = {'input': (self.batch, ) + (-1, ) * 3}
            return
        input_shapes = OrderedDict()
        for input_description in config_inputs:
            input_shapes[input_description['name']] = input_description.get('shape', (self.batch, ) + (-1, ) * 3)
        self._inputs = input_shapes

    @property
    def inputs(self):
        return self._inputs

    @property
    def batch(self):
        return self._batch

    @property
    def output_blob(self):
        return next(iter(self.output_names))

    def load_module(self, model_cls, module_args, module_kwargs, checkpoint=None, state_key=None, python_path=None,
                    init_method=None
    ):
        module_parts = model_cls.split(".")
        model_cls = module_parts[-1]
        model_path = ".".join(module_parts[:-1])
        with append_to_path(python_path):
            model_cls = getattr(importlib.import_module(model_path), model_cls)
            module = model_cls(*module_args, **module_kwargs)
            if init_method is not None:
                if hasattr(model_cls, init_method):
                    init_method = getattr(module, init_method)
                    module = init_method()
                else:
                    raise ValueError(f'Could not call the method {init_method} in the module {model_cls}.')

            if checkpoint:
                if isinstance(checkpoint, str) and re.match(CHECKPOINT_URL_REGEX, checkpoint):
                    checkpoint = urllib.request.urlretrieve(checkpoint)[0]  # nosec B310  # disable urllib-urlopen check
                checkpoint = self._torch.load(
                    checkpoint, map_location=None if self.cuda else self._torch.device('cpu')
                )
                state = checkpoint if not state_key else checkpoint[state_key]
                if all(key.startswith('module.') for key in state):
                    module = self._torch.nn.DataParallel(module)
                module.load_state_dict(state, strict=False)

            return self.prepare_module(module, model_cls)

    def load_tranformers_module(self, pretrained_name, python_path):
        with append_to_path(python_path):
            if isinstance(transformers, UnsupportedPackage):
                transformers.raise_error(self.__class__.__name__)

            model_class = getattr(transformers, self.tranformers_class)
            pretrained_model = python_path if python_path else pretrained_name
            module = model_class.from_pretrained(pretrained_model)

        return self.prepare_module(module, model_class)

    def prepare_module(self, module, model_class):
        module.to('cuda' if self.cuda else 'cpu')
        module.eval()

        if self.use_torch_compile:
            if hasattr(model_class, 'compile'):
                module.compile()
            module = self._torch.compile(module, **self.compile_kwargs)

        return module


    def _convert_to_tensor(self, value, precision):
        if isinstance(value, self._torch.Tensor):
            return value
        if precision is None:
            precision = np.float32

        return self._torch.from_numpy(value.astype(precision)).to(self.device)

    def fit_to_input(self, data, layer_name, layout, precision, template=None):

        if precision is None and layer_name in SCALAR_INPUTS:
            precision = np.int64

        if layer_name == 'input' and isinstance(data[0], dict):
            tensor_dict = {}
            for key, val in data[0].items():
                if isinstance(val, dict):
                    sub_tensor = {}
                    for k, value in val.items():
                        sub_tensor[k] = self._convert_to_tensor(value, precision)
                    tensor_dict[key] = sub_tensor
                else:
                    tensor_dict[key] = self._convert_to_tensor(val, precision)

            return tensor_dict

        data_shape = np.shape(data)

        if layout is not None and len(data_shape) == len(layout):
            data = np.transpose(data, layout)
        else:
            data = np.array(data)

        return self._convert_to_tensor(data, precision)

    def _convert_to_numpy(self, input_dict):
        numpy_dict = {}
        for key, value in input_dict.items():
            if isinstance(value, self._torch.Tensor):
                numpy_dict[key] = value.detach().cpu().numpy()
            else:
                numpy_dict[key] = value
        return numpy_dict

    def _value_to_numpy(self, value):
        if isinstance(value, self._torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return value
        if hasattr(value, 'numpy'):
            return value.numpy()
        return np.array(value)

    @staticmethod
    def _is_ultralytics_result(value):
        return hasattr(value, 'boxes')

    def _is_ultralytics_results(self, outputs):
        if self._is_ultralytics_result(outputs):
            return True
        return isinstance(outputs, (list, tuple)) and bool(outputs) and self._is_ultralytics_result(outputs[0])

    def _infer_num_classes(self, result):
        names = getattr(result, 'names', None)
        if isinstance(names, dict):
            return max(len(names), 1)
        if isinstance(names, (list, tuple)):
            return max(len(names), 1)

        boxes = getattr(result, 'boxes', None)
        if boxes is not None and hasattr(boxes, 'cls'):
            cls_ids = self._value_to_numpy(boxes.cls).astype(np.int64).reshape(-1)
            if cls_ids.size:
                return int(np.max(cls_ids)) + 1

        return 1

    def _ultralytics_result_to_numpy(self, result, num_classes):
        boxes = getattr(result, 'boxes', None)
        if boxes is None or not hasattr(boxes, 'xywh'):
            return np.zeros((4 + num_classes, 0), dtype=np.float32)

        xywh = self._value_to_numpy(boxes.xywh).astype(np.float32)
        if xywh.ndim == 1:
            xywh = np.expand_dims(xywh, 0)
        if xywh.size == 0:
            return np.zeros((4 + num_classes, 0), dtype=np.float32)

        num_boxes = xywh.shape[0]
        if hasattr(boxes, 'conf'):
            conf = self._value_to_numpy(boxes.conf).astype(np.float32).reshape(-1)
        else:
            conf = np.ones((num_boxes,), dtype=np.float32)
        if hasattr(boxes, 'cls'):
            cls_ids = self._value_to_numpy(boxes.cls).astype(np.int64).reshape(-1)
        else:
            cls_ids = np.zeros((num_boxes,), dtype=np.int64)

        if conf.size < num_boxes:
            conf = np.pad(conf, (0, num_boxes - conf.size), constant_values=1.0)
        if cls_ids.size < num_boxes:
            cls_ids = np.pad(cls_ids, (0, num_boxes - cls_ids.size), constant_values=0)
        conf = conf[:num_boxes]
        cls_ids = np.clip(cls_ids[:num_boxes], 0, num_classes - 1)

        class_scores = np.zeros((num_boxes, num_classes), dtype=np.float32)
        class_scores[np.arange(num_boxes), cls_ids] = conf

        return np.concatenate((xywh, class_scores), axis=1).T

    def _convert_ultralytics_results(self, outputs):
        results_list = outputs if isinstance(outputs, (list, tuple)) else [outputs]
        if not results_list:
            return np.zeros((0, 0, 0), dtype=np.float32)

        num_classes = max(self._infer_num_classes(result) for result in results_list)
        prepared = [self._ultralytics_result_to_numpy(result, num_classes) for result in results_list]

        max_boxes = max((item.shape[1] for item in prepared), default=0)
        batch = np.zeros((len(prepared), 4 + num_classes, max_boxes), dtype=np.float32)
        for idx, item in enumerate(prepared):
            if item.size:
                batch[idx, :, :item.shape[1]] = item

        return batch

    def _extract_ultralytics_raw_tensor(self, value):
        if isinstance(value, self._torch.Tensor):
            return value
        if isinstance(value, np.ndarray):
            return self._torch.from_numpy(value)

        if isinstance(value, dict):
            for key in ('one2many', 'one2one', 'preds', 'pred', 'output', 'outputs'):
                if key in value:
                    tensor = self._extract_ultralytics_raw_tensor(value[key])
                    if tensor is not None:
                        return tensor
            for item in value.values():
                tensor = self._extract_ultralytics_raw_tensor(item)
                if tensor is not None:
                    return tensor
            return None

        if isinstance(value, (list, tuple)):
            fallback = None
            for item in value:
                tensor = self._extract_ultralytics_raw_tensor(item)
                if tensor is None:
                    continue
                if tensor.ndim >= 3:
                    return tensor
                if fallback is None:
                    fallback = tensor
            return fallback

        return None

    def _convert_ultralytics_end2end_outputs(self, outputs):
        if not isinstance(outputs, (list, tuple)) or len(outputs) < 2 or not isinstance(outputs[1], dict):
            return None

        branches = outputs[1]
        branch = None
        if isinstance(branches, dict):
            if self.ultralytics_raw_branch == 'one2one':
                branch = branches.get('one2one')
            elif self.ultralytics_raw_branch == 'one2many':
                branch = branches.get('one2many')
            else:
                # Auto mode defaults to end2end inference branch.
                branch = branches.get('one2one') or branches.get('one2many')
        if branch is None and isinstance(branches, dict):
            branch = branches.get('one2one') or branches.get('one2many')
        if not isinstance(branch, dict):
            return None

        boxes = branch.get('boxes')
        scores = branch.get('scores')
        if not isinstance(boxes, self._torch.Tensor) or not isinstance(scores, self._torch.Tensor):
            return None

        # Decode head branch boxes to image-scale boxes when possible.
        yolo_wrapper = getattr(self.module, 'model', None)
        detect_head = getattr(yolo_wrapper, 'model', None)
        detect_head = detect_head[-1] if detect_head is not None and len(detect_head) else None
        if detect_head is not None and hasattr(detect_head, '_get_decode_boxes') and 'feats' in branch:
            decoded = detect_head._get_decode_boxes(branch)
            if isinstance(decoded, self._torch.Tensor):
                boxes = decoded

        # End2end branches provide xyxy boxes; convert to xywh to match yolov8_detection adapter expectations.
        if boxes.shape[1] == 4:
            x1y1 = boxes[:, :2, :]
            x2y2 = boxes[:, 2:4, :]
            c_xy = (x1y1 + x2y2) / 2
            wh = x2y2 - x1y1
            boxes = self._torch.cat((c_xy, wh), dim=1)

        if self.ultralytics_raw_scores_sigmoid:
            scores = scores.sigmoid()

        return self._torch.cat((boxes, scores), dim=1)

    def _convert_ultralytics_raw_outputs(self, outputs):
        tensor = self._convert_ultralytics_end2end_outputs(outputs)
        if tensor is None:
            tensor = self._extract_ultralytics_raw_tensor(outputs)
        if tensor is None:
            raise ValueError('Failed to extract raw tensor from Ultralytics output.')

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 4:
            tensor = tensor.reshape(tensor.shape[0], tensor.shape[1], -1)

        if tensor.ndim != 3:
            raise ValueError(f'Unexpected raw Ultralytics tensor rank: {tensor.ndim}')

        # Normalize to [batch, channels, boxes] expected by yolov8_detection adapter.
        if tensor.shape[1] > tensor.shape[2] and tensor.shape[2] <= 512:
            tensor = tensor.transpose(1, 2)

        return tensor.detach().cpu().numpy()

    def _extract_ultralytics_input_tensor(self, batch_input):
        if isinstance(batch_input, dict):
            if 'input' in batch_input and not isinstance(batch_input['input'], dict):
                candidate = batch_input['input']
            elif len(batch_input) == 1:
                candidate = next(iter(batch_input.values()))
            else:
                candidate = next(iter(batch_input.values()))
        else:
            candidate = batch_input

        if isinstance(candidate, np.ndarray):
            candidate = self._torch.from_numpy(candidate)
        if not isinstance(candidate, self._torch.Tensor):
            raise ValueError(f'Unsupported Ultralytics raw input type: {type(candidate)}')

        return candidate.to(self.device)

    def _predict_ultralytics_raw(self, batch_input):
        raw_model = getattr(self.module, 'model', None)
        if raw_model is None:
            raise ValueError('ultralytics_raw_output requires a module with .model attribute.')
        input_tensor = self._extract_ultralytics_input_tensor(batch_input)
        return raw_model(input_tensor)


    def forward(self, outputs):
        if hasattr(outputs, 'logits') and 'logits' in self.output_names:
            return {'logits': outputs.logits}
        if hasattr(outputs, 'last_hidden_state') and 'last_hidden_state' in self.output_names:
            return {'last_hidden_state': outputs.last_hidden_state}
        return list(outputs)

    def predict(self, inputs, metadata=None, **kwargs):
        results = []
        with self._torch.no_grad():
            for batch_input in inputs:
                if self.ultralytics_raw_output:
                    outputs = self._predict_ultralytics_raw(batch_input)
                    for meta_ in metadata:
                        if isinstance(batch_input, dict):
                            meta_['input_shape'] = {
                                key: list(data.shape) for key, data in batch_input.items() if hasattr(data, 'shape')
                            }
                elif metadata[0].get('input_is_dict_type') or (isinstance(batch_input, dict) and 'input' in batch_input):
                    outputs = self.module(batch_input['input'])
                else:
                    outputs = self.module(**batch_input)

                    for meta_ in metadata:
                        meta_['input_shape'] = {key: list(data.shape) for key, data in batch_input.items()}

                if self.ultralytics_raw_output:
                    result_dict = {self.output_names[0]: self._convert_ultralytics_raw_outputs(outputs)}
                elif metadata[0].get('output_is_dict_type') or isinstance(outputs, dict):
                    result_dict = self._convert_to_numpy(outputs)
                elif self._is_ultralytics_results(outputs):
                    result_dict = {self.output_names[0]: self._convert_ultralytics_results(outputs)}
                else:
                    result_dict = {
                        output_name: self._value_to_numpy(res)
                        for output_name, res in zip(self.output_names, outputs)
                    }
                results.append(result_dict)

        return results

    def predict_async(self, *args, **kwargs):
        raise ValueError('PyTorch Launcher does not support async mode yet')

    def release(self):
        del self.module


@contextmanager
def append_to_path(path):
    if path:
        sys.path.append(str(path))

    yield

    if path:
        sys.path.remove(str(path))
