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
from os import listdir
from os.path import join, isfile
from multiprocessing import Process, Value

import cv2
import numpy as np


class VideoLibrary:
    """ This class loads list of videos and plays each one in cycle. """

    def __init__(self, source_dir, max_size, class_names, visualizer_queue, trg_fps):
        """Constructor"""

        self.max_size = max_size

        self.source_paths = self.parse_source_paths(source_dir, class_names)
        assert len(self.source_paths) > 0, "Can't find videos in " + str(source_dir)

        self.cur_source_id = Value('i', 0, lock=True)

        self._visualizer_queue = visualizer_queue
        self._trg_time_step = 1. / float(trg_fps)
        self._play_process = None

    @property
    def num_sources(self):
        """Returns number of videos in the library"""

        return len(self.source_paths)

    @staticmethod
    def parse_source_paths(input_dir, valid_names):
        """Returns the list of valid video sources"""

        valid_names = set(n.lower() for n in valid_names)
        all_file_paths = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
        all_file_paths.sort()

        out_file_paths = []
        for file_path in all_file_paths:
            file_name = file_path.split('.')[0].lower()
            if file_name not in valid_names:
                continue

            full_file_path = join(input_dir, file_path)

            cap = cv2.VideoCapture(full_file_path)
            if not cap.isOpened():
                continue

            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if num_frames > 0:
                out_file_paths.append((file_name, full_file_path))

        return out_file_paths

    def next(self):
        """Moves pointer to the next video source"""

        with self.cur_source_id.get_lock():
            self.cur_source_id.value += 1
            if self.cur_source_id.value >= self.num_sources:
                self.cur_source_id.value = 0

    def prev(self):
        """Moves pointer to the previous video source"""

        with self.cur_source_id.get_lock():
            self.cur_source_id.value -= 1
            if self.cur_source_id.value < 0:
                self.cur_source_id.value = self.num_sources - 1

    def start(self):
        """Starts internal threads"""

        if self._play_process is not None and self._play_process.is_alive():
            return

        self._play_process = \
            Process(target=self._play,
                    args=(self._visualizer_queue, self.cur_source_id, self.source_paths,
                          self.max_size, self._trg_time_step))
        self._play_process.daemon = True
        self._play_process.start()

    def release(self):
        """Stops playing and releases internal storages"""

        if self._play_process is not None:
            self._play_process.terminate()
            self._play_process.join()

        self._play_process = None

    @staticmethod
    def _play(visualizer_queue, cur_source_id, source_paths, max_image_size, trg_time_step):
        """Produces live frame from the active video source"""

        cap = None
        last_source_id = cur_source_id.value

        while True:
            start_time = time.perf_counter()

            if cur_source_id.value != last_source_id:
                last_source_id = cur_source_id.value
                cap.release()
                cap = None

            source_name, source_path = source_paths[cur_source_id.value]

            if cap is None:
                cap = cv2.VideoCapture(source_path)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            _, frame = cap.read()
            if frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                _, frame = cap.read()
                assert frame is not None

            trg_frame_size = list(frame.shape[:2])
            if np.max(trg_frame_size) > max_image_size:
                if trg_frame_size[0] == np.max(trg_frame_size):
                    trg_frame_size[1] = int(float(max_image_size) / float(trg_frame_size[0]) * float(trg_frame_size[1]))
                    trg_frame_size[0] = max_image_size
                else:
                    trg_frame_size[0] = int(float(max_image_size) * float(trg_frame_size[0]) / float(trg_frame_size[1]))
                    trg_frame_size[1] = max_image_size

            frame = cv2.resize(frame, (trg_frame_size[1], trg_frame_size[0]))
            cv2.putText(frame, 'GT Gesture: {}'.format(source_name), (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            visualizer_queue.put(np.copy(frame), True)

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            rest_time = trg_time_step - elapsed_time
            if rest_time > 0.0:
                time.sleep(rest_time)
