"""
Copyright (c) 2024 Intel Corporation

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

import openvino_genai as ov_genai

from ...representation import CharacterRecognitionPrediction
from ...utils import UnsupportedPackage, extract_image_representations
from .base_custom_evaluator import BaseCustomEvaluator

try:
    import inflect
except ImportError as import_err:
    inflect = UnsupportedPackage("inflect", import_err.msg)


class WhisperEvaluator(BaseCustomEvaluator):
    def __init__(self, dataset_config, pipe, orig_config):
        super().__init__(dataset_config, None, orig_config)
        self.pipe = pipe
        if hasattr(self.pipe, 'adapter'):
            self.adapter_type = self.pipe.adapter.__provider__

    @classmethod
    def from_configs(cls, config, delayed_model_loading=False, orig_config=None):
        dataset_config = config['datasets']

        framework = config['launchers'][0]['framework']
        if framework == 'openvino':
            pipe = GenAI_WhisperPipeline(config)
        else:
            pipe = TransformersAsrPipeline(config)
        return cls(dataset_config, pipe, orig_config)

    def _process(self, output_callback, calculate_metrics, progress_reporter, metric_config, csv_file):
        for batch_id, (batch_input_ids, batch_annotation, batch_inputs, batch_identifiers) in enumerate(self.dataset):
            batch_inputs = self.preprocessor.process(batch_inputs, batch_annotation)
            batch_inputs_extr, batch_meta = extract_image_representations(batch_inputs)

            batch_raw_prediction, batch_prediction = self.pipe.predict(
                batch_identifiers, batch_inputs_extr, batch_meta
            )
            metrics_result = self._get_metrics_result(batch_input_ids, batch_annotation, batch_prediction,
                                                      calculate_metrics)
            if output_callback:
                output_callback(batch_raw_prediction[0], metrics_result=metrics_result,
                                element_identifiers=batch_identifiers, dataset_indices=batch_input_ids)
            self._update_progress(progress_reporter, metric_config, batch_id, len(batch_prediction), csv_file)

    def release(self):
        pass


def normalize_transcription(engine, text):
    # Convert numbers to words
    tokens = (engine.number_to_words(token) if token.isdigit() else token for token in text.split())
    # Remove punctuation except for apostrophes that are in the middle of words
    text = re.sub(r"\b'\b|[^\w\s]", '', ' '.join(tokens))
    # Remove leading, trailing, and multiple consecutive spaces, and convert to uppercase
    return ' '.join(text.upper().split())


class WhisperPipeline:
    def __init__(self, config):
        if isinstance(inflect,UnsupportedPackage):
            UnsupportedPackage("inflect", inflect.msg).raise_error(self.__class__.__name__)
        self.engine = inflect.engine()
        self.pipeline = self._initialize_pipeline(config)

    def _initialize_pipeline(self, config):
        raise NotImplementedError

    def _get_predictions(self, data, identifiers, input_meta):
        raise NotImplementedError

    def predict(self, identifiers, input_data, input_meta, encoder_callback=None):
        predictions = []
        outputs = []
        for data in input_data:
            transcription = self._get_predictions(data, identifiers, input_meta)
            prediction_text = normalize_transcription(self.engine, transcription)
            predictions.append(prediction_text)
            outputs.append(CharacterRecognitionPrediction(identifiers[0], predictions[0]))
        return [], outputs


class GenAI_WhisperPipeline(WhisperPipeline):
    def _initialize_pipeline(self, config):
        models_dirs = config.get('_models', [])
        device = config.get('_device', 'CPU')
        model_dir = models_dirs[0]
        pipeline = ov_genai.WhisperPipeline(str(model_dir), device=device)
        return pipeline

    def _get_predictions(self, data, identifiers, input_meta):
        return self.pipeline.generate(data[0]).texts[0]


class TransformersAsrPipeline(WhisperPipeline):
    def _initialize_pipeline(self, config):
        try:
            from transformers import (  # pylint: disable=C0415
                AutoModelForSpeechSeq2Seq, AutoProcessor)
            from transformers.pipelines.automatic_speech_recognition import \
                AutomaticSpeechRecognitionPipeline  # pylint: disable=C0415
        except ImportError as import_err:
            UnsupportedPackage("transformers", import_err.msg).raise_error(self.__class__.__name__)

        try:
            import torch  # pylint: disable=C0415
        except ImportError as import_err:
            UnsupportedPackage("torch", import_err.msg).raise_error(self.__class__.__name__)

        model_id = config.get('model_id')
        device = "cpu"

        # The following code is based on the implementation found at:
        # https://huggingface.co/openai/whisper-large-v3
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
        ).to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        pipeline = AutomaticSpeechRecognitionPipeline(
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )
        return pipeline

    def _get_predictions(self, data, identifiers, input_meta):
        sampling_rate = input_meta[0].get('sample_rate')
        sample = {'path': identifiers[0], 'array': data[0], 'sampling_rate': sampling_rate}
        return self.pipeline(sample)["text"]
