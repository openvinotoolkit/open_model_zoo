"""
Copyright (c) 2026 Intel Corporation

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

import importlib
import re

import numpy as np

from ...representation import CharacterRecognitionPrediction
from ...utils import UnsupportedPackage, extract_image_representations
from .base_custom_evaluator import BaseCustomEvaluator
from .whisper_evaluator import get_model_dir, normalize_transcription

try:
    import inflect
except ImportError as import_err:
    inflect = UnsupportedPackage("inflect", import_err.msg)


class ASRPipelineEvaluator(BaseCustomEvaluator):
    VALID_PIPELINE_CLASSES = [
        "GenAIASRPipeline",
        "HFASRPipeline",
        "HfFunAsrPipeline",
        "Qwen3ASROptimumPipeline",
        "FunAsrOptimumPipeline",
    ]

    def __init__(self, dataset_config, pipe, orig_config):
        super().__init__(dataset_config, None, orig_config)
        self.pipe = pipe
        if hasattr(self.pipe, "adapter"):
            self.adapter_type = self.pipe.adapter.__provider__

    @classmethod
    def from_configs(cls, config, delayed_model_loading=False, orig_config=None):
        dataset_config = config["datasets"]
        pipeline_class_name = config.get("pipeline_class", "GenAIASRPipeline")
        if "device" in config["launchers"][0]:
            config["_device"] = config["launchers"][0]["device"].upper()

        if pipeline_class_name not in cls.VALID_PIPELINE_CLASSES:
            raise ValueError(
                f"Invalid pipeline class name: {pipeline_class_name}. "
                f"Must be one of {cls.VALID_PIPELINE_CLASSES}"
            )

        pipeline_class = globals()[pipeline_class_name]
        pipe = pipeline_class(config)
        return cls(dataset_config, pipe, orig_config)

    def _process(
        self,
        output_callback,
        calculate_metrics,
        progress_reporter,
        metric_config,
        csv_file,
    ):
        for batch_id, (
            batch_input_ids,
            batch_annotation,
            batch_inputs,
            batch_identifiers,
        ) in enumerate(self.dataset):
            batch_inputs = self.preprocessor.process(batch_inputs, batch_annotation)
            batch_inputs_extr, batch_meta = extract_image_representations(batch_inputs)

            batch_raw_prediction, batch_prediction = self.pipe.predict(
                batch_identifiers, batch_inputs_extr, batch_meta
            )
            metrics_result = self._get_metrics_result(
                batch_input_ids, batch_annotation, batch_prediction, calculate_metrics
            )
            if output_callback:
                output_callback(
                    batch_raw_prediction[0],
                    metrics_result=metrics_result,
                    element_identifiers=batch_identifiers,
                    dataset_indices=batch_input_ids,
                )
            self._update_progress(
                progress_reporter,
                metric_config,
                batch_id,
                len(batch_prediction),
                csv_file,
            )

    def release(self):
        pass


class ASRPipeline:
    def __init__(self, config):
        self.engine = inflect.engine()
        self.pipeline = self._initialize_pipeline(config)

    def _initialize_pipeline(self, config):
        raise NotImplementedError

    def _get_predictions(self, data, identifier, input_meta):
        raise NotImplementedError

    def predict(self, identifiers, input_data, input_meta, encoder_callback=None):
        predictions = []
        outputs = []
        for identifier, data in zip(identifiers, input_data):
            data = self._normalize_audio(data)
            transcription = self._get_predictions(data, identifier, input_meta)
            prediction_text = normalize_transcription(self.engine, transcription)
            predictions.append(transcription)
            outputs.append(CharacterRecognitionPrediction(identifier, prediction_text))
        return predictions, outputs

    @staticmethod
    def _normalize_audio(data):
        data = np.asarray(data)
        if np.issubdtype(data.dtype, np.integer):
            return data.astype(np.float32) / -np.iinfo(data.dtype).min
        return data


class GenAIASRPipeline(ASRPipeline):
    def __init__(self, config):
        self.language = config.get("language")
        self.return_timestamps = config.get("return_timestamps", False)
        super().__init__(config)

    def _initialize_pipeline(self, config):
        try:
            import openvino_genai as ov_genai  # pylint: disable=C0415
        except ImportError as import_error:
            UnsupportedPackage("openvino_genai", import_error.msg).raise_error(
                self.__class__.__name__
            )

        model_dir = get_model_dir(config)
        device = config.get("_device", "CPU")
        return ov_genai.ASRPipeline(str(model_dir), device=device)

    def _get_predictions(self, data, identifier, input_meta):
        kwargs = {"return_timestamps": self.return_timestamps}
        if self.language:
            kwargs["language"] = self.language
        return self.pipeline.generate(data[0], **kwargs).texts[0]


class Qwen3ASROptimumPipeline(ASRPipeline):
    SAMPLE_RATE = 16000
    EOS_TOKEN_IDS = [151643, 151645]

    def __init__(self, config):
        self.max_new_tokens = config.get("max_new_tokens", 1000)
        self.language = config.get("language")
        super().__init__(config)

    def _initialize_pipeline(self, config):
        model_id = config.get("model_id")
        if "qwen" in str(model_id).lower() and "asr" in str(model_id).lower():
            try:
                importlib.import_module("qwen_asr")
            except ImportError as import_error:
                UnsupportedPackage("qwen-asr", import_error.msg).raise_error(
                    self.__class__.__name__
                )
        elif "fun" in str(model_id).lower() and "asr" in str(model_id).lower():
            try:
                importlib.import_module("fun_asr")
            except ImportError as import_error:
                UnsupportedPackage("fun-asr", import_error.msg).raise_error(
                    self.__class__.__name__
                )
        else:
            raise ValueError(f"Unsupported model_id: {model_id}")

        try:
            from optimum.intel.openvino import OVModelForSpeechSeq2Seq  # pylint: disable=C0415
        except ImportError as import_error:
            UnsupportedPackage("optimum.intel.openvino", import_error.msg).raise_error(
                self.__class__.__name__
            )

        try:
            from transformers import AutoProcessor  # pylint: disable=C0415
        except ImportError as import_error:
            UnsupportedPackage("transformers", import_error.msg).raise_error(
                self.__class__.__name__
            )

        model_dir = get_model_dir(config)
        device = config.get("_device", "CPU")
        ov_model = OVModelForSpeechSeq2Seq.from_pretrained(str(model_dir)).to(device)
        ov_processor = AutoProcessor.from_pretrained(str(model_dir))
        return ov_model, ov_processor

    def _get_predictions(self, data, identifier, input_meta):
        ov_model, ov_processor = self.pipeline
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": [{"type": "audio", "audio": ""}]},
        ]
        text_prompt = ov_processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        if self.language:
            text_prompt += f"language {self.language}<asr_text>"

        inputs = ov_processor(
            text=text_prompt,
            audio=data[0],
            sampling_rate=self.SAMPLE_RATE,
            return_tensors="pt",
        )

        output_ids = ov_model.generate(
            input_features=inputs["input_features"],
            decoder_input_ids=inputs["input_ids"],
            eos_token_id=self.EOS_TOKEN_IDS,
            max_new_tokens=self.max_new_tokens,
        )

        prompt_len = inputs["input_ids"].shape[1]
        generated_only = output_ids[:, prompt_len:]
        full_text = ov_processor.batch_decode(
            generated_only, skip_special_tokens=False
        )[0]
        return self.parse_asr_output(full_text)["text"]

    @staticmethod
    def parse_asr_output(raw_text):
        """Parse the raw ASR output to extract language and transcription text."""
        language_match = re.search(r"<\|([a-z]{2,3})\|>", raw_text)
        text_match = re.search(
            r"<asr_text>(.*?)(?:<\||$)", raw_text.replace("<|asr_text|>", "<asr_text>")
        )

        return {
            "language": language_match.group(1) if language_match else None,
            "text": text_match.group(1).strip() if text_match else raw_text.strip(),
        }


class FunAsrOptimumPipeline(ASRPipeline):
    SAMPLE_RATE = 16000
    EOS_TOKEN_IDS = [151643, 151645]

    def __init__(self, config):
        self.max_new_tokens = config.get("max_new_tokens", 1000)
        # FunASR prompt language name, e.g. "英文" (English), "中文" (Chinese)
        self.language = config.get("language", "英文")
        self.itn = config.get("itn", True)
        super().__init__(config)

    def _initialize_pipeline(self, config):
        try:
            from optimum.intel.openvino import OVModelForSpeechSeq2Seq  # pylint: disable=C0415
        except ImportError as import_error:
            UnsupportedPackage("optimum.intel.openvino", import_error.msg).raise_error(
                self.__class__.__name__
            )

        try:
            from transformers import AutoTokenizer  # pylint: disable=C0415
        except ImportError as import_error:
            UnsupportedPackage("transformers", import_error.msg).raise_error(
                self.__class__.__name__
            )

        model_dir = get_model_dir(config)
        device = config.get("_device", "CPU")
        ov_model = OVModelForSpeechSeq2Seq.from_pretrained(str(model_dir)).to(device)
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        return ov_model, tokenizer

    def _get_predictions(self, data, identifier, input_meta):
        ov_model, tokenizer = self.pipeline
        sampling_rate = input_meta[0].get("sample_rate", self.SAMPLE_RATE)
        audio = np.ascontiguousarray(data[0], dtype=np.float32).flatten()
        inputs = ov_model.preprocess_input(
            audio, sampling_rate=sampling_rate, language=self.language, itn=self.itn
        )

        output_ids = ov_model.generate(
            input_features=inputs["input_features"],
            decoder_input_ids=inputs["decoder_input_ids"],
            decoder_attention_mask=inputs["decoder_attention_mask"],
            eos_token_id=self.EOS_TOKEN_IDS,
            max_new_tokens=self.max_new_tokens,
        )

        prompt_len = inputs["decoder_input_ids"].shape[1]
        return tokenizer.batch_decode(
            output_ids[:, prompt_len:], skip_special_tokens=True
        )[0].strip()


class HFASRPipeline(ASRPipeline):
    def __init__(self, config):
        self.language = config.get("language")
        self.max_new_tokens = config.get("max_new_tokens", 1000)
        super().__init__(config)

    def _initialize_pipeline(self, config):
        model_id = config.get("model_id")

        if "qwen3" in str(model_id).lower() and "asr" in str(model_id).lower():
            try:
                from qwen_asr import Qwen3ASRModel  # pylint: disable=C0415
            except ImportError:
                Qwen3ASRModel = None

            if Qwen3ASRModel is not None:
                return Qwen3ASRModel.from_pretrained(
                    model_id,
                    max_inference_batch_size=1,
                    max_new_tokens=self.max_new_tokens,
                )

        try:
            import torch  # pylint: disable=C0415
        except ImportError as import_error:
            UnsupportedPackage("torch", import_error.msg).raise_error(self.__class__.__name__)

        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor  # pylint: disable=C0415
            from transformers.pipelines.automatic_speech_recognition import (  # pylint: disable=C0415
                AutomaticSpeechRecognitionPipeline,
            )
        except ImportError as import_error:
            UnsupportedPackage("transformers", import_error.msg).raise_error(
                self.__class__.__name__
            )

        device = config.get("_device", "cpu")
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
        ).to(device.lower())

        processor = AutoProcessor.from_pretrained(model_id)

        pipeline = AutomaticSpeechRecognitionPipeline(
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device.lower(),
        )
        return pipeline

    def _get_predictions(self, data, identifier, input_meta):
        sampling_rate = input_meta[0].get("sample_rate")
        if hasattr(self.pipeline, "transcribe"):
            result = self.pipeline.transcribe(
                (data[0], sampling_rate),
                context=identifier,
                language=self.language,
                return_time_stamps=False,
            )
            return result[0].text

        sample = {
            "path": identifier,
            "array": data[0],
            "sampling_rate": sampling_rate,
        }
        return self.pipeline(
            sample,
            return_timestamps=True,
            generate_kwargs={
                "num_beams": 1,
            },
        )["text"]

class HfFunAsrPipeline(ASRPipeline):
    def __init__(self, config):
        self.device = config.get("_device", "cpu").lower()
        # FunASR prompt language name, e.g. "英文" (English), "中文" (Chinese)
        self.language = config.get("language", "英文")
        self.itn = config.get("itn", True)
        super().__init__(config)

    def _initialize_pipeline(self, config):
        try:
            from funasr import AutoModel
        except ImportError as ie:
            UnsupportedPackage("funasr", ie.msg).raise_error(self.__class__.__name__)

        model_id = config.get("model_id")
        model = AutoModel(
            model=model_id,
            hub="hf",
            trust_remote_code=True,
            device=self.device
        )
        return model

    def _get_predictions(self, data, identifier, input_meta):
        import torch

        audio = torch.from_numpy(np.ascontiguousarray(data[0], dtype=np.float32)).flatten()
        res = self.pipeline.generate(
            input=[audio],
            cache={},
            hotwords=[],
            language=self.language,
            itn=self.itn,
        )
        return res[0]["text"]
