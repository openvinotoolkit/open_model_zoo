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
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from accuracy_checker.evaluators.custom_evaluators.whisper_evaluator import (
    GenAIWhisperPipeline, HFWhisperPipeline, OptimumWhisperPipeline,
    WhisperEvaluator)
from datasets import load_dataset

AutoProcessor = pytest.importorskip("transformers", reason="transformers is not available").AutoProcessor
AutoTokenizer = pytest.importorskip("transformers", reason="transformers is not available").AutoTokenizer
export_tokenizer = pytest.importorskip("optimum.exporters.openvino.convert", reason="optimum.exporters.openvino.convert is not available").export_tokenizer
OVModelForSpeechSeq2Seq = pytest.importorskip("optimum.intel.openvino", reason="optimum.intel.openvino is not available").OVModelForSpeechSeq2Seq


model_id = "openai/whisper-tiny"
model_dir = Path("/tmp/whisper-tiny")

def setup_module(module):
    # Setup code here
    global input_data, input_meta, identifiers

    # Load a single sample from the dataset
    dataset = load_dataset("openslr/librispeech_asr", "clean", split="validation", streaming=True, trust_remote_code=True)
    sample = next(iter(dataset))
    input_data = [sample["audio"]["array"]]
    input_meta = [{"sample_rate": sample["audio"]["sampling_rate"]}]
    identifiers = [sample["id"]]

def teardown_module(module):
    # Cleanup code here
    if model_dir.exists():
        for item in model_dir.iterdir():
            if item.is_file():
                item.unlink()
        model_dir.rmdir()

def test_optimum_convert_model_to_ir():
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    base_model = OVModelForSpeechSeq2Seq.from_pretrained(model_id)

    model_dir.mkdir(parents=True, exist_ok=True)
    base_model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    processor.save_pretrained(model_dir)
    export_tokenizer(tokenizer, model_dir)

    assert base_model.__class__.__module__.startswith('optimum.intel.openvino')

class TestWhisperEvaluator:
    def test_hf_whisper_pipeline(self):
        config = {"model_id": model_id}
        pipeline = HFWhisperPipeline(config)
        evaluator = WhisperEvaluator(None, pipeline, None)

        result = evaluator.pipe._get_predictions(input_data, identifiers, input_meta)
        assert isinstance(result, str)

    @pytest.mark.dependency(depends=["test_optimum_convert_model_to_ir"])
    def test_genai_whisper_pipeline(self):
        config = {"_models": [model_dir], "_device": "CPU"}
        pipeline = GenAIWhisperPipeline(config)
        evaluator = WhisperEvaluator(None, pipeline, None)

        result = evaluator.pipe._get_predictions(input_data, identifiers, input_meta)
        assert isinstance(result, str)

    @pytest.mark.dependency(depends=["test_optimum_convert_model_to_ir"])
    def test_optimum_whisper_pipeline(self):
        config = {"_models": [model_dir], "_device": "CPU"}
        pipeline = OptimumWhisperPipeline(config)
        evaluator = WhisperEvaluator(None, pipeline, None)

        result = evaluator.pipe._get_predictions(input_data, identifiers, input_meta)
        assert isinstance(result, str)
