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
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from accuracy_checker.evaluators.custom_evaluators.whisper_evaluator import (
    GenAIWhisperPipeline, OptimumWhisperPipeline, HFWhisperPipeline,
    WhisperEvaluator, normalize_transcription)
from datasets import load_dataset
from optimum.exporters.openvino.convert import export_tokenizer
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from transformers import AutoTokenizer, AutoProcessor

def export_model(model_id, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    base_model = OVModelForSpeechSeq2Seq.from_pretrained(model_id)

    base_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    export_tokenizer(tokenizer, output_dir)

model_id = "openai/whisper-tiny"
model_dir = Path("/tmp/whisper-tiny")

# Export the model
export_model(model_id, model_dir)

class TestWhisperEvaluator:
    @classmethod
    def setup_class(cls):
        # Load a single sample from the dataset
        dataset = load_dataset("openslr/librispeech_asr", "clean", split="validation", streaming=True, trust_remote_code=True)
        sample = next(iter(dataset))
        cls.input_data = [sample["audio"]["array"]]
        cls.input_meta = [{"sample_rate": sample["audio"]["sampling_rate"]}]
        cls.identifiers = [sample["id"]]

    @classmethod
    def teardown_class(cls):
        if model_dir.exists():
            for item in model_dir.iterdir():
                if item.is_file():
                    item.unlink()
            model_dir.rmdir()


    def test_hf_whisper_pipeline(self):
        config = {"model_id": model_id}
        pipeline = HFWhisperPipeline(config)
        evaluator = WhisperEvaluator(None, pipeline, None)

        result = evaluator.pipe._get_predictions(self.input_data, self.identifiers, self.input_meta)
        assert isinstance(result, str)

    def test_genai_whisper_pipeline(self):
        config = {"_models": [model_dir], "_device": "CPU"}
        pipeline = GenAIWhisperPipeline(config)
        evaluator = WhisperEvaluator(None, pipeline, None)

        result = evaluator.pipe._get_predictions(self.input_data, self.identifiers, self.input_meta)
        assert isinstance(result, str)

    def test_optimum_whisper_pipeline(self):
        config = {"_models": [model_dir], "_device": "CPU"}
        pipeline = OptimumWhisperPipeline(config)
        evaluator = WhisperEvaluator(None, pipeline, None)

        result = evaluator.pipe._get_predictions(self.input_data, self.identifiers, self.input_meta)
        assert isinstance(result, str)

def test_normalize_transcription():
    engine = MagicMock()
    engine.number_to_words.side_effect = lambda x: "one" if x == "1" else x
    text = "This is a test 1"
    result = normalize_transcription(engine, text)
    assert result == "THIS IS A TEST ONE"

