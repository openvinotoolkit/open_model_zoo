from collections import OrderedDict
import gc
import sys
import threading

import numpy as np

import accuracy_checker.representation.image_processing as image_processing
from accuracy_checker.representation.image_processing import ImageProcessingAnnotation


def test_image_cache_evicts_least_recently_used_above_half_of_ram(monkeypatch):
    monkeypatch.setattr(image_processing, '_get_total_memory', lambda: 100)
    cache = OrderedDict()
    monkeypatch.setattr(ImageProcessingAnnotation, '_cached_annotations', cache)
    monkeypatch.setattr(ImageProcessingAnnotation, '_cached_bytes', 0)

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
    assert ImageProcessingAnnotation._cached_bytes == 50


def test_image_cache_does_not_evict_pinned_annotations(monkeypatch):
    monkeypatch.setattr(image_processing, '_get_total_memory', lambda: 100)
    cache = OrderedDict()
    monkeypatch.setattr(ImageProcessingAnnotation, '_cached_annotations', cache)
    monkeypatch.setattr(ImageProcessingAnnotation, '_cached_bytes', 0)

    first = ImageProcessingAnnotation('first', 'first.png')
    second = ImageProcessingAnnotation('second', 'second.png')
    first.value = np.zeros(40, dtype=np.uint8)
    second.value = np.zeros(40, dtype=np.uint8)

    with image_processing.pinned_annotations([first]):
        assert ImageProcessingAnnotation.release_cache_if_needed()
        assert first._value is not None
        assert second._value is None

    assert first._pin_count == 0
    second.value = np.zeros(40, dtype=np.uint8)
    assert ImageProcessingAnnotation.release_cache_if_needed()
    assert first._value is None


def test_image_cache_copies_array_views(monkeypatch):
    cache = OrderedDict()
    monkeypatch.setattr(ImageProcessingAnnotation, '_cached_annotations', cache)
    monkeypatch.setattr(ImageProcessingAnnotation, '_cached_bytes', 0)

    full = np.arange(100, dtype=np.uint8).reshape(10, 10)
    crop = full[2:4, 3:6]
    annotation = ImageProcessingAnnotation('crop', 'crop.png')
    annotation.value = crop

    assert annotation._value.base is None
    np.testing.assert_array_equal(annotation._value, crop)
    assert sum(cache.values()) == ImageProcessingAnnotation._cached_bytes == crop.nbytes


def test_image_cache_byte_counter_tracks_cached_sizes(monkeypatch):
    cache = OrderedDict()
    monkeypatch.setattr(ImageProcessingAnnotation, '_cached_annotations', cache)
    monkeypatch.setattr(ImageProcessingAnnotation, '_cached_bytes', 0)

    first = ImageProcessingAnnotation('first', 'first.png')
    second = ImageProcessingAnnotation('second', 'second.png')
    first.value = np.zeros(10, dtype=np.uint8)
    second.value = np.zeros(20, dtype=np.uint8)
    assert ImageProcessingAnnotation._cached_bytes == sum(cache.values()) == 30

    first.value = np.zeros(15, dtype=np.uint8)
    assert ImageProcessingAnnotation._cached_bytes == sum(cache.values()) == 35

    first.release()
    assert ImageProcessingAnnotation._cached_bytes == sum(cache.values()) == 20

    del second
    gc.collect()
    assert ImageProcessingAnnotation._cached_bytes == sum(cache.values()) == 0


def test_image_cache_accounting_is_thread_safe(monkeypatch):
    monkeypatch.setattr(image_processing, '_get_total_memory', lambda: 100)
    cache = OrderedDict()
    monkeypatch.setattr(ImageProcessingAnnotation, '_cached_annotations', cache)
    monkeypatch.setattr(ImageProcessingAnnotation, '_cached_bytes', 0)
    switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    errors = []

    def worker():
        try:
            kept = []
            for i in range(500):
                annotation = ImageProcessingAnnotation(str(i), 'image.png')
                annotation.value = np.zeros(10, dtype=np.uint8)
                if i % 2:
                    kept.append(annotation)
                ImageProcessingAnnotation.release_cache_if_needed()
        except Exception as error:  # pylint: disable=broad-except
            errors.append(error)

    try:
        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        sys.setswitchinterval(switch_interval)

    gc.collect()
    assert not errors
    assert ImageProcessingAnnotation._cached_bytes == sum(cache.values())
    assert ImageProcessingAnnotation._cached_bytes <= 50
