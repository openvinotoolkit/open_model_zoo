from collections import OrderedDict

import numpy as np

import accuracy_checker.representation.image_processing as image_processing
from accuracy_checker.representation.image_processing import ImageProcessingAnnotation


def test_image_cache_evicts_least_recently_used_above_half_of_ram(monkeypatch):
    monkeypatch.setattr(image_processing, '_get_total_memory', lambda: 100)
    cache = OrderedDict()
    monkeypatch.setattr(ImageProcessingAnnotation, '_cached_annotations', cache)

    first = ImageProcessingAnnotation('first', 'first.png')
    second = ImageProcessingAnnotation('second', 'second.png')
    first.value = np.zeros(25, dtype=np.uint8)
    second.value = np.zeros(25, dtype=np.uint8)
    assert not ImageProcessingAnnotation.release_cache_if_needed()
    assert sum(cache.values()) == 50
    # Reading .value marks first as most recently used, making second the eviction candidate.
    assert first.value is not None

    third = ImageProcessingAnnotation('third', 'third.png')
    third.value = np.zeros(25, dtype=np.uint8)
    assert ImageProcessingAnnotation.release_cache_if_needed()
    assert first._value is not None
    assert second._value is None
    assert third._value is not None
    assert sum(cache.values()) == 50
