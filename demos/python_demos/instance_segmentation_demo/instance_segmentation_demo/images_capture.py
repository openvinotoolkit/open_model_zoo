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

from __future__ import print_function

import os
import os.path as osp

import cv2


class ImagesCapture(object):
    STATUS_OK = 1
    STATUS_FAIL = 0

    def __init__(self, path, skip_non_images=True):
        self.skip_non_images = skip_non_images
        self.images = []
        if osp.isdir(path):
            self.images = list(osp.join(path, i)
                               for i in sorted(os.listdir(path))
                               if osp.isfile(osp.join(path, i)))
        elif osp.isfile(path):
            self.images = [path, ]
        else:
            raise ValueError('"path" is neither an image file, not a directory with images.')
        self.pos = 0

    def isOpened(self):
        return True
    
    def release(self):
        pass

    def read(self):
        status = self.STATUS_FAIL
        image = None

        read_one_more_image = True
        while read_one_more_image:
            if self.pos >= len(self.images):
                break
            try:
                image_file_path = self.images[self.pos]
                self.pos += 1
                image = cv2.imread(image_file_path)
                status = self.STATUS_OK
                read_one_more_image = False
            except Exception:
                read_one_more_image = self.skip_non_images
                status = self.STATUS_FAIL
                image = None

        return status, image
