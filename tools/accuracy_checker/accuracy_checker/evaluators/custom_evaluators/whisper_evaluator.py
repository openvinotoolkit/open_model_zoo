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
import re
import os

from ...representation import CharacterRecognitionPrediction
from ...utils import UnsupportedPackage, extract_image_representations
from .base_custom_evaluator import BaseCustomEvaluator

try:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
except ImportError as import_err:
    AutoModelForSpeechSeq2Seq = UnsupportedPackage("transformers", import_err.msg)
    AutoProcessor = UnsupportedPackage("transformers", import_err.msg)

try:
    from transformers.pipelines.automatic_speech_recognition import AutomaticSpeechRecognitionPipeline
except ImportError as import_err:
    AutomaticSpeechRecognitionPipeline = UnsupportedPackage("transformers", import_err.msg)

try:
    import inflect
except ImportError as import_err:
    inflect = UnsupportedPackage("inflect", import_err.msg)


class WhisperEvaluator(BaseCustomEvaluator):
    VALID_PIPELINE_CLASSES = [
        "GenAIWhisperPipeline",
        "HFWhisperPipeline",
        "OptimumWhisperPipeline"
    ]

    def __init__(self, dataset_config, pipe, orig_config):
        super().__init__(dataset_config, None, orig_config)
        self.pipe = pipe
        if hasattr(self.pipe, "adapter"):
            self.adapter_type = self.pipe.adapter.__provider__

    @classmethod
    def from_configs(cls, config, delayed_model_loading=False, orig_config=None):
        dataset_config = config["datasets"]
        pipeline_class_name = config["pipeline_class"]
        if 'device' in config['launchers'][0]:
            config["_device"] = config['launchers'][0]['device'].upper()

        if pipeline_class_name not in cls.VALID_PIPELINE_CLASSES:
            raise ValueError(f"Invalid pipeline class name: {pipeline_class_name}. "
                             f"Must be one of {cls.VALID_PIPELINE_CLASSES}")

        pipeline_class = globals()[pipeline_class_name]
        pipe = pipeline_class(config)
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
    text = re.sub(r"\b'\b|[^\w\s]", "", " ".join(tokens))
    # Remove leading, trailing, and multiple consecutive spaces, and convert to uppercase
    return " ".join(text.upper().split())


class WhisperPipeline:
    def __init__(self, config):
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


class GenAIWhisperPipeline(WhisperPipeline):
    def _initialize_pipeline(self, config):
        try:
            import openvino_genai as ov_genai  # pylint: disable=C0415
        except ImportError as import_error:
            UnsupportedPackage("openvino_genai", import_error.msg).raise_error(self.__class__.__name__)

        model_dir = get_model_dir(config)
        device = config.get("_device", "CPU")
        pipeline = ov_genai.WhisperPipeline(str(model_dir), device=device)
        return pipeline

    def _get_predictions(self, data, identifiers, input_meta):
        return self.pipeline.generate(data[0], return_timestamps=True).texts[0]


class HFWhisperPipeline(WhisperPipeline):
    def _initialize_pipeline(self, config):
        try:
            import torch  # pylint: disable=C0415
        except ImportError as import_error:
            UnsupportedPackage("torch", import_error.msg).raise_error(self.__class__.__name__)

        model_id = config.get("model_id")
        device = "cpu"
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
        sampling_rate = input_meta[0].get("sample_rate")
        sample = {"path": identifiers[0], "array": data[0], "sampling_rate": sampling_rate}
        return self.pipeline(sample, return_timestamps=True)["text"]


class OptimumWhisperPipeline(WhisperPipeline):
    def _initialize_pipeline(self, config):
        try:
            from optimum.intel.openvino import OVModelForSpeechSeq2Seq  # pylint: disable=C0415
        except ImportError as import_error:
            UnsupportedPackage("optimum.intel.openvino", import_error.msg).raise_error(self.__class__.__name__)

        device = config.get("_device", "CPU")
        model_dir = get_model_dir(config)
        ov_model = OVModelForSpeechSeq2Seq.from_pretrained(str(model_dir)).to(device)
        ov_processor = AutoProcessor.from_pretrained(str(model_dir))

        pipeline = AutomaticSpeechRecognitionPipeline(
            model=ov_model,
            tokenizer=ov_processor.tokenizer,
            feature_extractor=ov_processor.feature_extractor
        )
        return pipeline

    def _get_predictions(self, data, identifiers, input_meta):
        sampling_rate = input_meta[0].get("sample_rate")
        sample = {"path": identifiers[0], "array": data[0], "sampling_rate": sampling_rate}
        return self.pipeline(sample, return_timestamps=True)["text"]



def get_model_dir(config):
    model_path = config.get("_models", [None])[0]

    if os.path.isfile(model_path):
        return os.path.dirname(model_path)
    return model_path
