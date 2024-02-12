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

import logging as log
from collections import Counter, defaultdict, deque
from functools import partial
from itertools import islice

import cv2
import numpy as np

from .meters import WindowAverageMeter

FONT_COLOR = (255, 255, 255)
FONT_STYLE = cv2.FONT_HERSHEY_DUPLEX
FONT_SIZE = 1
TEXT_VERTICAL_INTERVAL = 45
TEXT_LEFT_MARGIN = 15


class ResultRenderer:
    def __init__(self, no_show, presenter, output, limit, display_fps=False, display_confidence=True, number_of_predictions=1,
                 label_smoothing_window=30, labels=None, output_height=720):
        self.no_show = no_show
        self.presenter = presenter
        self.output = output
        self.limit = limit
        self.video_writer = cv2.VideoWriter()
        self.number_of_predictions = number_of_predictions
        self.display_confidence = display_confidence
        self.display_fps = display_fps
        self.labels = labels
        self.output_height = output_height
        self.meters = defaultdict(partial(WindowAverageMeter, 16))
        self.postprocessing = [LabelPostprocessing(n_frames=label_smoothing_window, history_size=label_smoothing_window)
                               for _ in range(number_of_predictions)]

    def update_timers(self, timers):
        inference_time = 0.0
        for key, val in timers.items():
            self.meters[key].update(val)
            inference_time += self.meters[key].avg
        return inference_time

    def render_frame(self, frame, logits, timers, frame_ind, raw_output, fps):
        inference_time = self.update_timers(timers)

        if logits is not None:
            labels, probs = decode_output(logits, self.labels, top_k=self.number_of_predictions,
                                          label_postprocessing=self.postprocessing)
            if raw_output:
                log.debug("Frame # {}: {} - {:.2f}% -- {:.2f}ms".format(frame_ind, labels[0], probs[0] * 100, inference_time))
        else:
            labels = ['Preparing...']
            probs = [0.]

        # resize frame, keep aspect ratio
        w, h, c = frame.shape
        new_h = self.output_height
        new_w = int(h * (new_h / w))
        frame = cv2.resize(frame, (new_w, new_h))

        self.presenter.drawGraphs(frame)
        # Fill text area
        fill_area(frame, (0, 70), (700, 0), alpha=0.6, color=(0, 0, 0))

        if self.display_confidence and logits is not None:
            text_template = '{label} - {conf:.2f}%'
        else:
            text_template = '{label}'

        for i, (label, prob) in enumerate(islice(zip(labels, probs), self.number_of_predictions)):
            display_text = text_template.format(label=label, conf=prob * 100)
            text_loc = (TEXT_LEFT_MARGIN, TEXT_VERTICAL_INTERVAL * (i + 1))

            cv2.putText(frame, display_text, text_loc, FONT_STYLE, FONT_SIZE, FONT_COLOR)

        if frame_ind == 0 and self.output and not self.video_writer.open(self.output,
            cv2.VideoWriter_fourcc(*'MJPG'), fps, (frame.shape[1], frame.shape[0])):
            log.error("Can't open video writer")
            return -1

        if self.display_fps:
            fps = 1000 / (inference_time + 1e-6)
            text_loc = (TEXT_LEFT_MARGIN, TEXT_VERTICAL_INTERVAL * (len(labels) + 1))
            cv2.putText(frame, "Inference time: {:.2f}ms ({:.2f} FPS)".format(inference_time, fps),
                        text_loc, FONT_STYLE, FONT_SIZE, FONT_COLOR)

        if self.video_writer.isOpened() and (self.limit <= 0 or frame_ind <= self.limit-1):
            self.video_writer.write(frame)

        if not self.no_show:
            cv2.imshow("Action Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in {ord('q'), ord('Q'), 27}:
                return -1
            self.presenter.handleKey(key)


class LabelPostprocessing:
    def __init__(self, n_frames=5, history_size=30):
        self.n_frames = n_frames
        self.history = deque(maxlen=history_size)
        self.prev_get = None
        self.prev_label = None

    def update(self, label):
        self.prev_label = label
        self.history.append(label)

    def get(self):
        if self.prev_get is None:
            self.prev_get = self.prev_label
            return self.prev_label

        cnt = Counter(list(self.history)[-self.n_frames:])
        if len(cnt) > 1:
            return self.prev_get
        self.prev_get = self.prev_label
        return self.prev_get


def fill_area(image, bottom_left, top_right, color=(0, 0, 0), alpha=1.):
    """Fills area with the specified color"""
    xmin, ymax = bottom_left
    xmax, ymin = top_right

    image[ymin:ymax, xmin:xmax, :] = image[ymin:ymax, xmin:xmax, :] * (1 - alpha) + np.asarray(color) * alpha
    return image


def decode_output(probs, labels, top_k=None, label_postprocessing=None):
    """Decodes top probabilities into corresponding label names"""
    top_ind = np.argsort(probs)[::-1][:top_k]

    if label_postprocessing:
        for k in range(top_k):
            label_postprocessing[k].update(top_ind[k])

        top_ind = [postproc.get() for postproc in label_postprocessing]

    decoded_labels = [labels[i] if labels else str(i) for i in top_ind]
    probs = [probs[i] for i in top_ind]
    return decoded_labels, probs
