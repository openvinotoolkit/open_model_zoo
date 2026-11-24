from unittest.mock import MagicMock

import pytest

from accuracy_checker.evaluators import quantization_model_evaluator
from accuracy_checker.evaluators.quantization_model_evaluator import ModelEvaluator


class _SyncInferQueue:
    def __init__(self):
        self._callback = None

    def set_callback(self, callback):
        self._callback = callback

    def start_async(self, request, user_data):
        self._callback(request, user_data)

    def wait_all(self):
        pass


def _create_evaluator():
    launcher = MagicMock()
    launcher.predict.return_value = ([], [])
    launcher.get_result_from_request.return_value = ([], [])
    launcher.get_infer_queue.return_value = _SyncInferQueue()
    launcher.prepare_data_for_request.side_effect = (
        lambda inputs, meta, batch_id, input_ids, annotation, identifiers:
        (None, (batch_id, input_ids, annotation, identifiers, meta))
    )
    evaluator = ModelEvaluator(launcher, None, {})
    evaluator.dataset = [([0], [], [], ['a']), ([1], [], [], ['b'])]
    evaluator._prepare_to_evaluation = MagicMock()
    evaluator._get_batch_input = MagicMock(return_value=([], [], None))
    evaluator._switch_to_sync = MagicMock(return_value=False)
    evaluator._set_number_infer_requests = MagicMock()
    return evaluator


@pytest.mark.parametrize('method', ['process_dataset', 'process_dataset_async'])
def test_image_cache_limit_enforced_per_batch(monkeypatch, method):
    release = MagicMock()
    monkeypatch.setattr(
        quantization_model_evaluator.ImageProcessingAnnotation, 'release_cache_if_needed', release
    )

    getattr(_create_evaluator(), method)(calculate_metrics=False)

    assert release.call_count == 2
