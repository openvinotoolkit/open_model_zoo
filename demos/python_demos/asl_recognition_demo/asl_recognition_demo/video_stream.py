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
from os.path import exists
from multiprocessing import Process, Value, Array

import cv2
import numpy as np


class VideoStream:
    """ This class returns constant framerate frames from the input stream. """

    def __init__(self, input_source, trg_fps, batch_size):
        """Constructor"""

        try:
            self._input_source = int(input_source)
        except ValueError:
            self._input_source = input_source

        self._trg_fps = trg_fps
        assert self._trg_fps > 0
        self._batch_size = batch_size
        assert self._batch_size > 0

        cap = cv2.VideoCapture(self._input_source)
        assert cap.isOpened(), "Can't open " + str(self._input_source)

        source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()

        self._image_shape = [source_height, source_width, 3]
        self._batch_shape = [batch_size] + self._image_shape

        self._image_buffer_size = int(np.prod(self._image_shape))
        self._batch_buffer_size = int(np.prod(self._batch_shape))

        self._source_finished = Value('i', False, lock=False)
        self._raw_frame = Array('B', self._image_buffer_size, lock=True)
        self._slow_frame = Array('B', self._image_buffer_size, lock=True)
        self._slow_batch = Array('B', self._batch_buffer_size, lock=True)

        self._frame_generator_process = None
        self._producer_process = None

    def get_live_frame(self):
        """Returns last live frame from the input stream"""

        if self._source_finished.value:
            return None

        with self._raw_frame.get_lock():
            buffer = np.frombuffer(self._raw_frame.get_obj(), dtype=np.uint8)
            frame = np.copy(buffer.reshape(self._image_shape))

        return frame

    def get_batch(self):
        """Returns last batch of frames with constant framerate from the input stream"""

        if self._source_finished.value:
            return None

        with self._slow_batch.get_lock():
            buffer = np.frombuffer(self._slow_batch.get_obj(), dtype=np.uint8)
            batch = np.copy(buffer.reshape(self._batch_shape))

        return batch

    def start(self):
        """Starts internal threads"""

        self._frame_generator_process = \
            Process(target=self._frame_generator,
                    args=(self._input_source, self._raw_frame, self._image_shape,
                          self._source_finished))
        self._frame_generator_process.daemon = True
        self._frame_generator_process.start()

        self._producer_process = \
            Process(target=self._producer,
                    args=(self._raw_frame, self._slow_frame, self._slow_batch,
                          self._trg_fps, self._batch_size, self._image_shape,
                          self._source_finished))
        self._producer_process.daemon = True
        self._producer_process.start()

    def release(self):
        """Release internal threads"""

        if self._frame_generator_process is not None:
            self._frame_generator_process.terminate()
            self._frame_generator_process.join()

        if self._producer_process is not None:
            self._producer_process.terminate()
            self._producer_process.join()

    @staticmethod
    def _frame_generator(input_source, out_frame, frame_shape, finish_flag):
        """Produces live frames from the input stream"""

        cap = cv2.VideoCapture(input_source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        source_fps = cap.get(cv2.CAP_PROP_FPS)
        trg_time_step = 1.0 / float(source_fps)

        while True:
            start_time = time.perf_counter()

            _, frame = cap.read()
            if frame is None:
                break

            with out_frame.get_lock():
                buffer = np.frombuffer(out_frame.get_obj(), dtype=np.uint8)
                np.copyto(buffer.reshape(frame_shape), frame)

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            rest_time = trg_time_step - elapsed_time
            if rest_time > 0.0:
                time.sleep(rest_time)

        finish_flag.value = True
        cap.release()

    @staticmethod
    def _producer(input_frame, out_frame, out_batch, trg_fps, batch_size, image_shape, finish_flag):
        """Produces frames and batch of frames with constant framerate
           from the internal stream of frames"""

        trg_time_step = 1.0 / float(trg_fps)
        batch_shape = [batch_size] + image_shape
        frame_buffer = []

        while not finish_flag.value:
            start_time = time.perf_counter()

            with input_frame.get_lock():
                in_frame_buffer = np.frombuffer(input_frame.get_obj(), dtype=np.uint8)
                frame = np.copy(in_frame_buffer.reshape(image_shape))

            with out_frame.get_lock():
                out_frame_buffer = np.frombuffer(out_frame.get_obj(), dtype=np.uint8)
                np.copyto(out_frame_buffer.reshape(image_shape), frame)

            frame_buffer.append(frame)
            if len(frame_buffer) > batch_size:
                frame_buffer = frame_buffer[-batch_size:]

            if len(frame_buffer) == batch_size:
                with out_batch.get_lock():
                    out_batch_buffer = np.frombuffer(out_batch.get_obj(), dtype=np.uint8)
                    np.copyto(out_batch_buffer.reshape(batch_shape), frame_buffer)

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            rest_time = trg_time_step - elapsed_time
            if rest_time > 0.0:
                time.sleep(rest_time)
