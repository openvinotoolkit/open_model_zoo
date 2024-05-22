"""
 Copyright (c) 2020-2024 Intel Corporation

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

import time
from collections import deque

import cv2
import numpy as np

from .meters import MovingAverageMeter
from .models import AsyncWrapper, preprocess_frame
from .pipeline import AsyncPipeline, PipelineStep
from .queue import Signal


def run_pipeline(capture, model_type, model, render_fn, raw_output, seq_size=16, fps=30):
    pipeline = AsyncPipeline()
    pipeline.add_step("Data", DataStep(capture), parallel=False)

    if model_type in ('en-de', 'en-mean'):
        pipeline.add_step("Encoder", EncoderStep(model[0]), parallel=False)
        pipeline.add_step("Decoder", DecoderStep(model[1], sequence_size=seq_size), parallel=False)
    elif model_type == 'i3d-rgb':
        pipeline.add_step("I3DRGB", I3DRGBModelStep(model[0], seq_size, 256, 224), parallel=False)

    pipeline.add_step("Render", RenderStep(render_fn, raw_output, fps=fps), parallel=True)

    pipeline.run()
    pipeline.close()
    pipeline.print_statistics()


def softmax(x, axis=None):
    """Normalizes logits to get confidence values along specified axis"""
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis)


class I3DRGBModelStep(PipelineStep):
    def __init__(self, model, sequence_size, frame_size, crop_size):
        super().__init__()
        self.model = model
        assert sequence_size > 0
        self.sequence_size = sequence_size
        self.size = frame_size
        self.crop_size = crop_size
        self.input_seq = deque(maxlen = self.sequence_size)
        self.async_model = AsyncWrapper(self.model, self.model.num_requests)

    def process(self, frame):
        preprocessed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preprocessed = preprocess_frame(preprocessed, self.size, self.crop_size, chw_layout=False)
        self.input_seq.append(preprocessed)
        if len(self.input_seq) == self.sequence_size:
            input_blob = np.array(self.input_seq)
            input_blob = np.expand_dims(input_blob, axis=0)
            output, next_frame = self.async_model.infer(input_blob, frame)

            if output is None:
                return None

            return next_frame, output[0], {'i3d-rgb-model': self.own_time.last}

        return frame, None, {'i3d-rgb-model': self.own_time.last}


class DataStep(PipelineStep):
    def __init__(self, capture):
        super().__init__()
        self.cap = capture

    def setup(self):
        pass

    def process(self, item):
        frame = self.cap.read()
        if frame is None:
            return Signal.STOP
        return frame

    def end(self):
        pass


class EncoderStep(PipelineStep):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.async_model = AsyncWrapper(self.encoder, self.encoder.num_requests)

    def __del__(self):
        self.encoder.cancel()

    def process(self, frame):
        preprocessed = preprocess_frame(frame)
        preprocessed = preprocessed[np.newaxis, ...]  # add batch dimension
        embedding, frame = self.async_model.infer(preprocessed, frame)

        if embedding is None:
            return None

        return frame, embedding.reshape((1, -1)), {'encoder': self.own_time.last}


class DecoderStep(PipelineStep):
    def __init__(self, decoder, sequence_size=16):
        super().__init__()
        assert sequence_size > 0
        self.sequence_size = sequence_size
        self.decoder = decoder
        self.async_model = AsyncWrapper(self.decoder, self.decoder.num_requests)
        self._embeddings = deque(maxlen=self.sequence_size)

    def __del__(self):
        self.decoder.cancel()

    def process(self, item):
        if item is None:
            return None

        frame, embedding, timers = item
        timers['decoder'] = self.own_time.last
        self._embeddings.append(embedding)

        if len(self._embeddings) == self.sequence_size:
            decoder_input = np.concatenate(self._embeddings, axis=0)
            decoder_input = np.expand_dims(decoder_input, axis=0)

            logits, next_frame = self.async_model.infer(decoder_input, frame)

            if logits is None:
                return None

            probs = softmax(logits - np.max(logits))
            return next_frame, probs[0], timers

        return frame, None, timers




class RenderStep(PipelineStep):
    """Passes inference result to render function"""

    def __init__(self, render_fn, raw_output, fps):
        super().__init__()
        self.render = render_fn
        self.raw_output = raw_output
        self.fps = fps
        self._frames_processed = 0
        self._t0 = None
        self._render_time = MovingAverageMeter(0.9)

    def process(self, item):
        if item is None:
            return
        self._sync_time()
        render_start = time.time()
        status = self.render(*item, self._frames_processed, self.raw_output, self.fps)
        self._render_time.update(time.time() - render_start)

        self._frames_processed += 1
        if status is not None and status < 0:
            return Signal.STOP_IMMEDIATELY
        return status

    def end(self):
        cv2.destroyAllWindows()

    def _sync_time(self):
        now = time.time()
        if self._t0 is None:
            self._t0 = now
        expected_time = self._t0 + (self._frames_processed + 1) / self.fps
        if self._render_time.avg:
            expected_time -= self._render_time.avg
        if expected_time > now:
            time.sleep(expected_time - now)
