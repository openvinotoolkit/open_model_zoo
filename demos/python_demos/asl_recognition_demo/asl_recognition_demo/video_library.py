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

from os import listdir
from os.path import join, isfile

import cv2


class VideoLibrary:
    """ This class loads list of videos and plays each one in cycle. """

    def __init__(self, source_dir, trg_size, class_names):
        """Constructor"""

        self.trg_size = trg_size

        self.source_paths = self.parse_source_paths(source_dir, class_names)
        assert len(self.source_paths) > 0, "Can't find videos in " + str(source_dir)

        self.cur_source_id = 0
        self.cap = None

    @property
    def num_sources(self):
        """Returns number of videos in the library"""

        return len(self.source_paths)

    @property
    def cur_source(self):
        """Returns the path to the current video source"""

        return self.source_paths[self.cur_source_id]

    @staticmethod
    def parse_source_paths(input_dir, valid_names):
        """Returns the list of valid video sources"""

        valid_names = [n.lower() for n in valid_names]
        all_file_paths = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]

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

    def release(self):
        """Release internal storages"""

        if self.cap is not None:
            self.cap.release()

        self.cap = None

    def next(self):
        """Moves pointer to the next video source"""

        self.release()

        self.cur_source_id += 1
        if self.cur_source_id >= self.num_sources:
            self.cur_source_id = 0

    def prev(self):
        """Moves pointer to the previous video source"""

        self.release()

        self.cur_source_id -= 1
        if self.cur_source_id < 0:
            self.cur_source_id = self.num_sources - 1

    def get_frame(self):
        """Returns current frame from the active video source"""

        source_name, source_path = self.cur_source

        if self.cap is None:
            self.cap = cv2.VideoCapture(source_path)

        _, frame = self.cap.read()
        if frame is None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            _, frame = self.cap.read()
            assert frame is not None

        if frame is not None:
            frame = cv2.resize(frame, self.trg_size)

            cv2.putText(frame, 'Gesture: {}'.format(source_name), (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame
