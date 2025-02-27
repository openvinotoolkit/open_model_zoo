"""
Copyright (c) 2024-2025 Intel Corporation

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
import os
import numpy as np
from PIL import Image
from scipy.special import softmax

from .base_custom_evaluator import BaseCustomEvaluator
from .base_models import BaseCascadeModel
from ...config import ConfigError
from ...utils import contains_all, extract_image_representations, read_json, UnsupportedPackage
from ...representation import ClassificationPrediction
from ...logging import print_info

try:
    from tqdm import tqdm
except ImportError as error:
    tqdm = UnsupportedPackage('tqdm', error.msg)

try:
    import open_clip
except ImportError as clip_error:
    open_clip = UnsupportedPackage('open_clip', clip_error.msg)

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError as transformers_error:
    AutoModel = UnsupportedPackage('AutoModel', transformers_error.msg)
    AutoTokenizer = UnsupportedPackage('AutoTokenizer', transformers_error.msg)

try:
    import torch
except ImportError as torch_error:
    torch = UnsupportedPackage("torch", torch_error.msg)

class OpenVinoClipEvaluator(BaseCustomEvaluator):
    def __init__(self, dataset_config, launcher, model, orig_config):
        super().__init__(dataset_config, launcher, orig_config)
        self.model = model

    @classmethod
    def from_configs(cls, config, delayed_model_loading=False, orig_config=None):
        dataset_config, launcher, _ = cls.get_dataset_and_launcher_info(config)
        model = OpenVinoClipVitModel(
            config.get('network_info', {}), launcher, config.get('_models', []),
            config.get('_model_is_blob'),
            delayed_model_loading, config
        )
        return cls(dataset_config, launcher, model, orig_config)

    def _process(self, output_callback, calculate_metrics, progress_reporter, metric_config, csv_file):

        zeroshot_weights = self.model.zero_shot_classifier(self.dataset.data_reader.data_source)
        for batch_id, (batch_input_ids, batch_annotation, batch_inputs, batch_identifiers) in enumerate(self.dataset):
            batch_inputs = self.preprocessor.process(batch_inputs, batch_annotation)
            batch_data, _ = extract_image_representations(batch_inputs)

            batch_raw_prediction, batch_prediction = self.model.predict(
                batch_identifiers, batch_data, zeroshot_weights
            )

            metrics_result = self._get_metrics_result(batch_input_ids, batch_annotation, batch_prediction,
                                                      calculate_metrics)
            if output_callback:
                output_callback(batch_raw_prediction, metrics_result=metrics_result,
                                element_identifiers=batch_identifiers, dataset_indices=batch_input_ids)
            self._update_progress(progress_reporter, metric_config, batch_id, len(batch_prediction), csv_file)


class OpenVinoJinaClipEvaluator(OpenVinoClipEvaluator):
    @classmethod
    def from_configs(cls, config, delayed_model_loading=False, orig_config=None):
        if config['launchers'][0]['framework'] == 'pytorch':
            dataset_config, launcher = config["datasets"], None
            delayed_model_loading = False
        else:
            dataset_config, launcher, _ = cls.get_dataset_and_launcher_info(config)

        model = OpenVinoJinaClipModel(
            config.get('network_info', {}), launcher, config.get('_models', []),
            config.get('_model_is_blob'),
            delayed_model_loading, config
        )
        return cls(dataset_config, launcher, model, orig_config)


class BaseOpenVinoClipModel(BaseCascadeModel):
    def __init__(self, network_info, launcher, models_args, is_blob, delayed_model_loading=False, config=None):
        super().__init__(network_info, launcher, delayed_model_loading)
        self.network_info = network_info
        self.launcher = launcher
        self.config = config or {}
        self.templates_file = None
        self.parameters_file = None
        self.templates = ["a photo of a {classname}"]
        self.parts = network_info.keys()
        if launcher:
            network_info = self.fill_part_with_model(network_info, self.parts,
                                                                 models_args, False, delayed_model_loading)
            if not contains_all(network_info, self.parts) and not delayed_model_loading:
                raise ConfigError('configuration for text_encoder/image_encoder does not exist')
        if not delayed_model_loading:
            self.create_pipeline(launcher, network_info)

    def create_pipeline(self, launcher, network_info):
        raise NotImplementedError("Subclasses should implement this method")

    def get_logits(self, image_features, zeroshot_weights):
        raise NotImplementedError("Subclasses should implement this method")

    def predict(self, identifiers, input_data, zeroshot_weights):
        preds = []
        for idx, image_data in zip(identifiers, input_data):
            image_features = self.encode_image(image_data)
            logits = self.get_logits(image_features, zeroshot_weights)
            preds.append(ClassificationPrediction(idx, np.squeeze(logits, axis=0)))
        return None, preds

    def get_network(self):
        models = self.pipe.get_models()
        model_list = []
        for model_part_name, model in models.items():
            model_list.append({"name": model_part_name, "model": model})
        return model_list

    def encode_image(self, image_data):
        raise NotImplementedError("Subclasses should implement this method")

    def encode_text(self, texts, params):
        raise NotImplementedError("Subclasses should implement this method")

    @staticmethod
    def get_pretrained_model_params(path):
        params = {}
        open_clip_params = np.load(path)
        params['attn_mask'] = open_clip_params['attn_mask']
        params['token_embedding'] = open_clip_params['token_embedding']
        params['positional_embedding'] = open_clip_params['positional_embedding']
        params['text_projection'] = open_clip_params['text_projection']
        params['normalized_shape'] = open_clip_params['normalized_shape']
        params['gamma'] = open_clip_params['gamma']
        params['beta'] = open_clip_params['beta']
        return params

    def get_class_embeddings(self, texts, params):
        raise NotImplementedError("Subclasses should implement this method")

    def zero_shot_classifier(self, data_source):
        classnames = read_json(os.path.join(data_source, self.classnames_file))
        if self.templates_file:
            templates = read_json(os.path.join(data_source, self.templates_file))
        else:
            templates = self.templates

        params = None
        if self.parameters_file:
            params = self.get_pretrained_model_params(os.path.join(data_source, self.parameters_file))

        print_info('Encoding zeroshot weights for {} imagenet classes'.format(len(classnames)))

        zeroshot_weights = []
        iterator = classnames
        if not isinstance(tqdm, UnsupportedPackage):
            iterator = tqdm(classnames, mininterval=2)

        for classname in iterator:
            texts = [template.format(classname=classname) for template in templates]
            class_embeddings = self.get_class_embeddings(texts, params)
            zeroshot_weights.append(class_embeddings)
        return np.stack(zeroshot_weights, axis=1)

    def load_models(self, network_info, launcher, log=False):
        if isinstance(network_info, dict):
            for model_name, model_dict in network_info.items():
                model_dict["name"] = model_name
                self.load_model(model_dict, launcher)
        else:
            for model_dict in network_info:
                self.load_model(model_dict, launcher)

        if log:
            self.print_input_output_info()

    def load_model(self, network_list, launcher):
        model, weights = self.automatic_model_search(network_list)
        if weights:
            network = launcher.read_network(str(model), str(weights))
        else:
            network = launcher.read_network(str(model), None)
        setattr(self, "{}_model".format(network_list["name"]), network)

    def print_input_output_info(self):
        model_parts = self.parts
        for part in model_parts:
            part_model_id = "{}_model".format(part)
            model = getattr(self, part_model_id, None)
            if model is not None:
                self.launcher.print_input_output_info(model, part)

    @staticmethod
    def layer_norm(input_array, gamma, beta, epsilon=1e-5):
        """
        Input array layer normalization (aka torch.nn.LayerNorm).
        """
        mean = np.mean(input_array, axis=-1, keepdims=True)
        std = np.std(input_array, axis=-1, keepdims=True)
        normalized = (input_array - mean) / np.sqrt(std ** 2 + epsilon)
        return normalized * gamma + beta

    @staticmethod
    def normalize(input_array, p=2, axis=-1, epsilon=1e-12):
        """
        Input array normalization using the p-norm (aka torch.nn.functional.normalize).
        """
        norm = np.linalg.norm(input_array, ord=p, axis=axis, keepdims=True)
        norm = np.maximum(norm, epsilon)
        normalized = input_array / norm
        return normalized


class OpenVinoClipVitModel(BaseOpenVinoClipModel):
    def create_pipeline(self, launcher, network_info):
        orig_model_name = self.config.get("orig_model_name", "ViT-B-16-plus-240")
        self.load_models(network_info, launcher, True)
        self.text_encoder = launcher.ie_core.compile_model(self.text_encoder_model, launcher.device)
        self.image_encoder = launcher.ie_core.compile_model(self.image_encoder_model, launcher.device)
        unet_shapes = [inp.get_partial_shape() for inp in self.text_encoder_model.inputs]
        if unet_shapes[0][0].is_dynamic:
            self.templates_file = self.config.get("templates", "zeroshot_classification_templates.json")

        self.classnames_file = self.config.get("classnames", "classnames.json")
        self.parameters_file = self.config.get("pretrained_model_params", None)
        self.tokenizer = open_clip.get_tokenizer(orig_model_name)

    def get_logits(self, image_features, zeroshot_weights):
        image_features = self.normalize(image_features, axis=-1)
        logits = 100. * image_features @ zeroshot_weights
        return logits

    def encode_image(self, image_data):
        image = np.expand_dims(image_data, axis=0)
        features = self.image_encoder(image)
        return features[self.image_encoder.output()]

    def encode_text(self, texts, params):
        text = self.tokenizer(texts).to('cpu')
        indices = text.detach().cpu().numpy()

        x = params['token_embedding'][indices]
        x = x + params['positional_embedding']
        x = x.transpose(1, 0, 2)
        x = self.text_encoder((x, params['attn_mask']))
        x = x[self.text_encoder.output()]
        x = x.transpose(1, 0, 2)
        x = self.layer_norm(x, params['gamma'], params['beta'])
        x = x[np.arange(x.shape[0]), np.argmax(indices, axis=-1)] @ params['text_projection']
        return x

    def get_class_embeddings(self, texts, params):
        class_embeddings = self.encode_text(texts, params)
        class_embedding = self.normalize(class_embeddings, axis=-1)
        class_embedding = np.mean(class_embedding, axis=0)
        class_embedding /= np.linalg.norm(class_embedding, ord=2)
        return class_embedding


class OpenVinoJinaClipModel(BaseOpenVinoClipModel):
    def create_pipeline(self, launcher, network_info):
        if isinstance(AutoTokenizer, UnsupportedPackage):
            AutoTokenizer.raise_error(self.__class__.__name__)
        if isinstance(AutoModel, UnsupportedPackage):
            AutoModel.raise_error(self.__class__.__name__)
        if isinstance(torch, UnsupportedPackage):
            torch.raise_error(self.__class__.__name__)

        orig_model_name = self.config.get("orig_model_name", "jinaai/jina-clip-v1")

        model = AutoModel.from_pretrained(orig_model_name, trust_remote_code=True)
        if launcher:
            self.load_models(network_info, launcher, True)
            self.text_encoder = launcher.ie_core.compile_model(self.text_model, launcher.device)
            self.vision_encoder = launcher.ie_core.compile_model(self.vision_model, launcher.device)
        else:
            self.text_encoder = model.text_model
            self.vision_encoder = model.vision_model

        self.templates = ["{classname}"]
        self.classnames_file = self.config.get("classnames", "classnames.json")
        self.tokenizer = AutoTokenizer.from_pretrained(orig_model_name, trust_remote_code=True)
        self.processor = model.get_preprocess()

    def encode_image(self, image_data):
        image = Image.fromarray(image_data)
        vision_input = self.processor(images=[image], return_tensors="pt")
        image_embeddings = self.vision_encoder(vision_input["pixel_values"])

        if isinstance(image_embeddings, torch.Tensor):
            image_embeddings = image_embeddings.detach().numpy()
        else:
            image_embeddings = image_embeddings[0]

        return image_embeddings

    def encode_text(self, text_input):
        text_embeddings = self.text_encoder(text_input["input_ids"])

        if isinstance(text_embeddings, torch.Tensor):
            text_embeddings = text_embeddings.detach().numpy()
        else:
            text_embeddings = text_embeddings[0]
        return text_embeddings

    def get_logits(self, image_features, zeroshot_weights):
        text_embeddings = np.squeeze(zeroshot_weights)
        simularity = []
        for emb1 in image_features:
            temp_simularity = []
            for emb2 in text_embeddings:
                temp_simularity.append(emb1 @ emb2)
            simularity.append(temp_simularity)

        logits = 100. * softmax(simularity)
        return logits

    def get_class_embeddings(self, texts, params):
        text_input = self.tokenizer(texts, return_tensors="pt", padding="max_length",
                                    max_length=512, truncation=True).to("cpu")
        return self.encode_text(text_input)
