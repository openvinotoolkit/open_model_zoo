"""
 Copyright (c) 2019 Intel Corporation

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
from itertools import cycle

import cv2
import numpy as np

from .meters import MovingAverageMeter
from .models import AsyncWrapper, preprocess_frame
from .pipeline import AsyncPipeline, PipelineStep
from .queue import Signal


def run_pipeline(video, encoder, decoder, render_fn, fps=30):
    pipeline = AsyncPipeline()
    pipeline.add_step("Data", DataStep(video), parallel=False)
    pipeline.add_step("Encoder", EncoderStep(encoder), parallel=False)
    pipeline.add_step("Decoder", DecoderStep(decoder), parallel=False)
    pipeline.add_step("Render", RenderStep(render_fn, fps=fps), parallel=True)

    pipeline.run()
    pipeline.close()
    pipeline.print_statistics()


class DataStep(PipelineStep):

    def __init__(self, video_list, loop=True):
        super().__init__()
        self.video_list = video_list
        self.cap = None

        if loop:
            self._video_cycle = cycle(self.video_list)
        else:
            self._video_cycle = iter(self.video_list)

    def setup(self):
        self._open_video()

    def process(self, item):
        if not self.cap.isOpened() and not self._open_video():
            return Signal.STOP
        status, frame = self.cap.read()
        if not status:
            return Signal.STOP
        return frame

    def end(self):
        self.cap.release()

    def _open_video(self):
        next_video = next(self._video_cycle)
        try:
            next_video = int(next_video)
        except ValueError:
            pass
        self.cap = cv2.VideoCapture(next_video)
        if not self.cap.isOpened():
            return False
        return True


class EncoderStep(PipelineStep):

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.async_model = AsyncWrapper(self.encoder, self.encoder.num_requests)

    def process(self, frame):
        preprocessed = preprocess_frame(frame)
        embedding, frame = self.async_model.infer(preprocessed, frame)

        if embedding is None:
            return None

        return frame, embedding.reshape((1, -1)), {'encoder': self.own_time.last}


class DecoderStep(PipelineStep):

    def __init__(self, decoder, sequence_size=16, num_classes=9):
        super().__init__()
        self.sequence_size = sequence_size
        self.num_classes = num_classes
        self.decoder = decoder
        self.async_model = AsyncWrapper(self.decoder, self.decoder.num_requests)
        self._embeddings = deque(maxlen=self.sequence_size)

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


def softmax(x, axis=None):
    """Normalizes logits to get confidence values along specified axis"""
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis)


class RenderStep(PipelineStep):
    """Passes inference result to render function"""

    def __init__(self, render_fn, fps):
        super().__init__()
        self.render = render_fn
        self.fps = fps
        self._frames_processed = 0
        self._t0 = None
        self._render_time = MovingAverageMeter(0.9)

    def process(self, item):
        if item is None:
            return
        self._sync_time()
        # status = None
        render_start = time.time()
        status = self.render(*item, self._frames_processed)
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
