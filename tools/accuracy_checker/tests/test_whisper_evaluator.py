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
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from accuracy_checker.evaluators.custom_evaluators.whisper_evaluator import (
    GenAIWhisperPipeline, OptimumWhisperPipeline, HFWhisperPipeline,
    WhisperEvaluator, normalize_transcription)
from datasets import load_dataset
from optimum.exporters.openvino.convert import export_tokenizer
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from transformers import AutoTokenizer,AutoProcessor


def export_model(model_id, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    base_model = OVModelForSpeechSeq2Seq.from_pretrained(model_id)

    base_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    export_tokenizer(tokenizer, output_dir)

model_name = "openai/whisper-tiny"
model_dir = Path("/tmp/whisper-tiny")

# Export the model
export_model(model_name, model_dir)

# Load a single sample from the dataset
dataset = load_dataset("openslr/librispeech_asr", "clean", split="validation", streaming=True, trust_remote_code=True)
sample = next(iter(dataset))
ground_truth = sample["text"]
input_data = [sample["audio"]["array"]]
input_meta = [{"sample_rate": sample["audio"]["sampling_rate"]}]
identifiers = [sample["id"]]
# print(ground_truth)

class TestWhisperEvaluator:
    def test_hf_whisper_pipeline(self):
        config = {"model_id": model_name}
        pipeline = HFWhisperPipeline(config)
        evaluator = WhisperEvaluator(None, pipeline, None)

        result = evaluator.pipe._get_predictions(input_data, identifiers, input_meta)
        assert isinstance(result, str)
        # print(result)

    def test_genai_whisper_pipeline(self):
        config = {"_models": [model_dir], "_device": "CPU"}
        pipeline = GenAIWhisperPipeline(config)
        evaluator = WhisperEvaluator(None, pipeline, None)

        result = evaluator.pipe._get_predictions(input_data, identifiers, input_meta)
        assert isinstance(result, str)
        # print(result)

    def test_optimum_whisper_pipeline(self):
        config = {"_models": [model_dir], "_device": "CPU"}
        pipeline = OptimumWhisperPipeline(config)
        evaluator = WhisperEvaluator(None, pipeline, None)

        result = evaluator.pipe._get_predictions(input_data, identifiers, input_meta)
        assert isinstance(result, str)
        # print(result)


def test_normalize_transcription():
    engine = MagicMock()
    engine.number_to_words.side_effect = lambda x: "one" if x == "1" else x
    text = "This is a test 1"
    result = normalize_transcription(engine, text)
    assert result == "THIS IS A TEST ONE"
