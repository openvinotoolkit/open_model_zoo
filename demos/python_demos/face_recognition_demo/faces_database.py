"""
 Copyright (c) 2018 Intel Corporation

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
import os
import os.path as osp
import cv2
import numpy as np

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine
from face_detector import FaceDetector

class FacesDatabase:
    IMAGE_EXTENSIONS = ['jpg', 'png']

    class Identity:
        def __init__(self, label, descriptors):
            self.label = label
            self.descriptors = descriptors

        @staticmethod
        def cosine_dist(x, y):
            return cosine(x, y) * 0.5

    def __init__(self, path, face_identifier, landmarks_detector, face_detector=None, no_show=False):
        path = osp.abspath(path)
        self.fg_path = path
        self.no_show = no_show
        paths = []
        if osp.isdir(path):
            paths = [osp.join(path, f) for f in os.listdir(path) \
                      if f.split('.')[-1] in self.IMAGE_EXTENSIONS]
        else:
            log.error("Wrong face images database path. Expected a " \
                      "path to the directory containing %s files, " \
                      "but got '%s'" % \
                      (" or ".join(self.IMAGE_EXTENSIONS), path))

        if len(paths) == 0:
            log.error("The images database folder has no images.")

        self.database = []
        for num, path in enumerate(paths):
            label = osp.splitext(osp.basename(path))[0]
            image = cv2.imread(path, flags=cv2.IMREAD_COLOR)

            assert len(image.shape) == 3, \
                "Expected an input image in (H, W, C) format"
            assert image.shape[2] in [3, 4], \
                "Expected BGR or BGRA input"

            orig_image = image.copy()
            image = image.transpose((2, 0, 1)) # HWC to CHW
            image = np.expand_dims(image, axis=0)

            if face_detector:
                face_detector.start_async(image)
                rois = face_detector.get_roi_proposals(image)
                if len(rois) < 1:
                    log.warning("Not found faces on the image '%s'" % (path))
            else:
                w, h = image.shape[-1], image.shape[-2]
                rois = [FaceDetector.Result([0, 0, 0, 0, 0, w, h])]

            for i, roi in enumerate(rois):
                r = [roi]
                landmarks_detector.start_async(image, r)
                landmarks = landmarks_detector.get_landmarks()

                face_identifier.start_async(image, r, landmarks)
                descriptor = face_identifier.get_descriptors()[0]

                if face_detector:
                    mm = self.check_if_face_exist(descriptor, face_identifier.get_threshold())
                    if mm < 0:
                        crop = orig_image[int(roi.position[1]):int(roi.position[1]+roi.size[1]), \
                               int(roi.position[0]):int(roi.position[0]+roi.size[0])]
                        name = self.ask_to_save(crop)
                        self.dump_faces(crop, descriptor, name)
                else:
                    log.debug("Adding label {} to the gallery.".format(label))
                    self.add_item(descriptor, label)

    def ask_to_save(self, image):
        if self.no_show:
            return None
        save = False
        label = None
        winname = "Unknown face"
        cv2.namedWindow(winname)
        cv2.moveWindow(winname, 0, 0)
        w = int(400 * image.shape[0]/image.shape[1])
        sz = (400, w)
        resized = cv2.resize(image, sz, interpolation=cv2.INTER_AREA)
        font = cv2.FONT_HERSHEY_PLAIN
        fontScale = 1
        fontColor = (255, 255, 255)
        lineType = 1
        img = cv2.copyMakeBorder(resized, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(img, 'This is an unrecognized image.', (30, 50), font, fontScale, fontColor, lineType)
        cv2.putText(img, 'If you want to store it to the gallery,', (30, 80), font, fontScale, fontColor, lineType)
        cv2.putText(img, 'please, put the name and press "Enter".', (30, 110), font, fontScale, fontColor, lineType)
        cv2.putText(img, 'Otherwise, press "Escape".', (30, 140), font, fontScale, fontColor, lineType)
        cv2.putText(img, 'You can see the name here:', (30, 170), font, fontScale, fontColor, lineType)
        name = ''
        while 1:
            cc = img.copy()
            cv2.putText(cc, name, (30, 200), font, fontScale, fontColor, lineType)
            cv2.imshow(winname, cc)

            k = cv2.waitKey(0)
            if k == 27: #Esc
                break
            if k == 13: #Enter
                if len(name) > 0:
                    save = True
                    break
                else:
                    cv2.putText(cc, "Name was not inserted. Please try again.", (30, 200), font, fontScale, fontColor, lineType)
                    cv2.imshow(winname, cc)
                    k = cv2.waitKey(0)
                    if k == 27:
                        break
                    continue
            if k == 225: #Shift
                continue
            if k == 8: #backspace
                name = name[:-1]
                continue
            else:
                name += chr(k)
                continue

        cv2.destroyWindow(winname)
        label = name if save else None
        return label

    def match_faces(self, descriptors):
        database = self.database
        distances = np.empty((len(descriptors), len(database)))
        for i, desc in enumerate(descriptors):
            for j, identity in enumerate(database):
                dist = []
                for k, id_desc in enumerate(identity.descriptors):
                    dist.append(FacesDatabase.Identity.cosine_dist(desc, id_desc))
                distances[i][j] = dist[np.argmin(dist)]

        # Find best assignments, prevent repeats, assuming faces can not repeat
        _, assignments = linear_sum_assignment(distances)
        matches = []
        for i in range(len(descriptors)):
            if len(assignments) <= i: # assignment failure, too many faces
                matches.append((0, 1.0))
                continue

            id = assignments[i]
            distance = distances[i, id]
            matches.append((id, distance))
        return matches

    def create_new_label(self, path, id):
        while osp.exists(osp.join(path, "face{}.jpg".format(id))):
            id += 1
        return "face{}".format(id)

    def check_if_face_exist(self, desc, threshold):
        match = -1
        for j, identity in enumerate(self.database):
            dist = []
            for k, id_desc in enumerate(identity.descriptors):
                dist.append(FacesDatabase.Identity.cosine_dist(desc, id_desc))
            if dist[np.argmin(dist)] < threshold:
                match = j
                break
        return match

    def check_if_label_exists(self, label):
        match = -1
        import re
        name = re.split(r'-\d+$', label)
        if not len(name):
            return -1, label
        label = name[0].lower()

        for j, identity in enumerate(self.database):
            if identity.label == label:
                match = j
                break
        return match, label

    def dump_faces(self, image, desc, name):
        match, label = self.add_item(desc, name)
        if match < 0:
            filename = "{}-0.jpg".format(label)
            match = len(self.database)-1
        else:
            filename = "{}-{}.jpg".format(label, len(self.database[match].descriptors)-1)
        filename = osp.join(self.fg_path, filename)

        log.debug("Dumping image with label {} and path {} on disk.".format(label, filename))
        if osp.exists(filename):
            log.warning("File with the same name already exists at {}. So it won't be stored.".format(self.fg_path))
        cv2.imwrite(filename, image)
        return match

    def add_item(self, desc, label):
        match = -1
        if not label:
            label = self.create_new_label(self.fg_path, len(self.database))
            log.warning("Trying to store an item without a label. Assigned label {}.".format(label))
        else:
            match, label = self.check_if_label_exists(label)

        if match < 0:
            self.database.append(FacesDatabase.Identity(label, [desc]))
            log.debug("Adding label {} to the database".format(label))
        else:
            self.database[match].descriptors.append(desc)
            log.debug("Appending new descriptor for label {}.".format(label))

        log.debug("The database length is {}.".format(len(self.database)))
        return match, label

    def __getitem__(self, idx):
        return self.database[idx]

    def __len__(self):
        return len(self.database)
