#!/usr/bin/env python
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
import math
import os
import os.path as osp
import sys
import time
from argparse import ArgumentParser

import cv2
import numpy as np
from numpy import clip

from openvino.inference_engine import IENetwork, IEPlugin
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine

DEVICE_KINDS = ['CPU', 'GPU', 'FPGA', 'MYRIAD', 'HETERO']


def build_argparser():
    parser = ArgumentParser()

    general = parser.add_argument_group('General')
    general.add_argument('-i', '--input', metavar="PATH", default='cam',
        help="(optional) Path to the input video " \
            "('cam' for the camera, default)")
    general.add_argument('-o', '--output', metavar="PATH", default="",
        help="(optional) Path to save the output video to")
    general.add_argument('--no_show', action='store_true',
        help="(optional) Do not display output")
    general.add_argument('-tl', '--timelapse', action='store_true',
        help="(optional) Auto-pause after each frame")
    general.add_argument('-cw', '--crop_width', default=0, type=int,
        help="(optional) Crop the input stream to this width " \
            "(default: no crop). Both -cw and -ch parameters " \
            "should be specified to use crop.")
    general.add_argument('-ch', '--crop_height', default=0, type=int,
        help="(optional) Crop the input stream to this height " \
            "(default: no crop). Both -cw and -ch parameters " \
            "should be specified to use crop.")

    gallery = parser.add_argument_group('Faces database')
    gallery.add_argument('-fg', metavar="PATH", required=True,
        help="Path to the face images directory")
    gallery.add_argument('--run_detector', action='store_true',
        help="(optional) Use Face Detection model to find faces" \
            " on the face images, otherwise use full images.")

    models = parser.add_argument_group('Models')
    models.add_argument('-m_fd', metavar="PATH", default="", required=True,
        help="Path to the Face Detection model XML file")
    models.add_argument('-m_lm', metavar="PATH", default="", required=True,
        help="Path to the Facial Landmarks Regression model XML file")
    models.add_argument('-m_reid', metavar="PATH", default="", required=True,
        help="Path to the Face Reidentification model XML file")

    infer = parser.add_argument_group('Inference options')
    infer.add_argument('-d_fd', default='CPU', choices=DEVICE_KINDS,
        help="(optional) Target device for the " \
            "Face Detection model (default: %(default)s)")
    infer.add_argument('-d_lm', default='CPU', choices=DEVICE_KINDS,
        help="(optional) Target device for the " \
            "Facial Landmarks Regression model (default: %(default)s)")
    infer.add_argument('-d_reid', default='CPU', choices=DEVICE_KINDS,
        help="(optional) Target device for the " \
            "Face Reidentification model (default: %(default)s)")
    infer.add_argument('-l', '--cpu_lib', metavar="PATH", default="",
        help="(optional) For MKLDNN (CPU)-targeted custom layers, if any. " \
            "Path to a shared library with custom layers implementations")
    infer.add_argument('-c', '--gpu_lib', metavar="PATH", default="",
        help="(optional) For clDNN (GPU)-targeted custom layers, if any. " \
            "Path to the XML file with descriptions of the kernels")
    infer.add_argument('-v', '--verbose', action='store_true',
        help="(optional) Be more verbose")
    infer.add_argument('-pc', '--perf_stats', action='store_true',
        help="(optional) Output detailed per-layer performance stats")
    infer.add_argument('-t_fd', metavar='[0..1]', type=float, default=0.6,
        help="(optional) Probability threshold for face detections" \
            "(default: %(default)s)")
    infer.add_argument('-t_id', metavar='[0..1]', type=float, default=0.3,
        help="(optional) Cosine distance threshold between two vectors " \
            "for face identification (default: %(default)s)")
    infer.add_argument('-exp_r_fd', metavar='NUMBER', type=float, default=1.15,
        help="(optional) Scaling ratio for bboxes passed to face recognition " \
            "(default: %(default)s)")
    infer.add_argument('--allow_grow', action='store_true',
        help="(optional) Allow to grow faces gallery and to dump on disk")

    return parser


class InferenceContext:
    def __init__(self):
        self.plugins = {}

    def load_plugins(self, devices, cpu_ext="", gpu_ext=""):
        log.info("Loading plugins for devices: %s" % (devices))

        plugins = { d: IEPlugin(d) for d in devices }
        if 'CPU' in plugins and not len(cpu_ext) == 0:
            log.info("Using CPU extensions library '%s'" % (cpu_ext))
            assert osp.isfile(cpu_ext), "Failed to open CPU extensions library"
            plugins['CPU'].add_cpu_extension(cpu_ext)

        if 'GPU' in plugins and not len(gpu_ext) == 0:
            assert osp.isfile(gpu_ext), "Failed to open GPU definitions file"
            plugins['GPU'].set_config({"CONFIG_FILE": gpu_ext})

        self.plugins = plugins

        log.info("Plugins are loaded")

    def get_plugin(self, device):
        return self.plugins.get(device, None)

    def check_model_support(self, net, device):
        plugin = self.plugins[device]

        if plugin.device == "CPU":
            supported_layers = plugin.get_supported_layers(net)
            not_supported_layers = [l for l in net.layers.keys() \
                                    if l not in supported_layers]
            if len(not_supported_layers) != 0:
                log.error("The following layers are not supported " \
                    "by the plugin for the specified device {}:\n {}" \
                    .format(plugin.device, ', '.join(not_supported_layers)))
                log.error("Please try to specify cpu extensions " \
                    "library path in the command line parameters using " \
                    "the '-l' parameter")
                raise NotImplementedError(
                    "Some layers are not supported on the device")

    def deploy_model(self, model, device, max_requests=1):
        self.check_model_support(model, device)
        plugin = self.plugins[device]
        deployed_model = plugin.load(network=model, num_requests=max_requests)
        return deployed_model


def cut_roi(frame, roi):
    p1 = roi.position.astype(int)
    p1 = clip(p1, [0, 0], [frame.shape[-1], frame.shape[-2]])
    p2 = (roi.position + roi.size).astype(int)
    p2 = clip(p2, [0, 0], [frame.shape[-1], frame.shape[-2]])
    return np.array(frame[ :, :, p1[1]:p2[1], p1[0]:p2[0] ])

def cut_rois(frame, rois):
    return [cut_roi(frame, roi) for roi in rois]

def resize_input(frame, target_shape):
    assert len(frame.shape) == len(target_shape), \
        "Expected a frame with %s dimensions, but got %s" % \
        (len(target_shape), len(frame.shape))

    assert frame.shape[0] == 1, "Only batch size 1 is supported"
    n, c, h, w = target_shape

    input = frame[0]
    if not np.array_equal(target_shape[-2:], frame.shape[-2:]):
        input = input.transpose((1, 2, 0)) # to HWC
        input = cv2.resize(input, (w, h))
        input = input.transpose((2, 0, 1)) # to CHW

    return input.reshape((n, c, h, w))


class Module(object):
    def __init__(self, model):
        self.model = model
        self.device_model = None

        self.max_requests = 0
        self.active_requests = 0

        self.clear()

    def deploy(self, device, context, queue_size=1):
        self.context = context
        self.max_requests = queue_size
        self.device_model = context.deploy_model(
            self.model, device, self.max_requests)
        self.model = None

    def enqueue(self, input):
        self.clear()

        if self.max_requests <= self.active_requests:
            log.warning("Processing request rejected - too many requests")
            return False

        self.device_model.start_async(self.active_requests, input)
        self.active_requests += 1
        return True

    def wait(self):
        if self.active_requests <= 0:
            return

        self.perf_stats = [None, ] * self.active_requests
        self.outputs = [None, ] * self.active_requests
        for i in range(self.active_requests):
            self.device_model.requests[i].wait()
            self.outputs[i] = self.device_model.requests[i].outputs
            self.perf_stats[i] = self.device_model.requests[i].get_perf_counts()

        self.active_requests = 0

    def get_outputs(self):
        self.wait()
        return self.outputs

    def get_performance_stats(self):
        return self.perf_stats

    def clear(self):
        self.perf_stats = []
        self.outputs = []


class FaceDetector(Module):
    class Result:
        OUTPUT_SIZE = 7

        def __init__(self, output):
            self.image_id = output[0]
            self.label = int(output[1])
            self.confidence = output[2]
            self.position = np.array((output[3], output[4])) # (x, y)
            self.size = np.array((output[5], output[6])) # (w, h)

        def _rescale_roi(self, roi_scale_factor=1.0):
            self.position -= self.size * 0.5 * (roi_scale_factor - 1.0)
            self.size *= roi_scale_factor

        def _resize_roi(self, frame_width, frame_height):
            self.position[0] *= frame_width
            self.position[1] *= frame_height
            self.size[0] = self.size[0] * frame_width - self.position[0]
            self.size[1] = self.size[1] * frame_height - self.position[1]

        def _clip(self, width, height):
            min = [0, 0]
            max = [width, height]
            self.position[:] = clip(self.position, min, max)
            self.size[:] = clip(self.size, min, max)

    def __init__(self, model, confidence_threshold=0.5, roi_scale_factor=1.15):
        super(FaceDetector, self).__init__(model)

        assert len(model.inputs) == 1, "Expected 1 input blob"
        assert len(model.outputs) == 1, "Expected 1 output blob"
        self.input_blob = next(iter(model.inputs))
        self.output_blob = next(iter(model.outputs))
        self.input_shape = model.inputs[self.input_blob].shape
        self.output_shape = model.outputs[self.output_blob].shape

        assert len(self.output_shape) == 4 and \
               self.output_shape[3] == self.Result.OUTPUT_SIZE, \
            "Expected model output shape with %s outputs" % \
            (self.Result.OUTPUT_SIZE)

        assert 0.0 <= confidence_threshold and confidence_threshold <= 1.0, \
            "Confidence threshold is expected to be in range [0; 1]"
        self.confidence_threshold = confidence_threshold

        assert 0.0 < roi_scale_factor, "Expected positive ROI scale factor"
        self.roi_scale_factor = roi_scale_factor

    def preprocess(self, frame):
        assert len(frame.shape) == 4, "Frame shape should be [1, c, h, w]"
        assert frame.shape[0] == 1
        assert frame.shape[1] == 3
        input = resize_input(frame, self.input_shape)
        return input

    def start_async(self, frame):
        input = self.preprocess(frame)
        self.enqueue(input)

    def enqueue(self, input):
        return super(FaceDetector, self).enqueue({self.input_blob: input})

    def get_roi_proposals(self, frame):
        outputs = self.get_outputs()[0][self.output_blob]
        # outputs shape is [N_requests, 1, 1, N_max_faces, 7]

        frame_width = frame.shape[-1]
        frame_height = frame.shape[-2]

        results = []
        for output in outputs[0][0]:
            result = FaceDetector.Result(output)
            if result.confidence < self.confidence_threshold:
                break # results are sorted by confidence decrease

            result._clip(1, 1)
            result._resize_roi(frame_width, frame_height)
            result._rescale_roi(self.roi_scale_factor)
            result._clip(frame_width, frame_height)

            results.append(result)

        return results


class LandmarksDetector(Module):
    POINTS_NUMBER = 5

    class Result:
        def __init__(self, outputs):
            self.points = outputs

            p = lambda i: self[i]
            self.left_eye = p(0)
            self.right_eye = p(1)
            self.nose_tip = p(2)
            self.left_lip_corner = p(3)
            self.right_lip_corner = p(4)

        def __getitem__(self, idx):
            return self.points[idx]

        def get_array(self):
            return np.array(self.points, dtype=np.float64)

    def __init__(self, model):
        super(LandmarksDetector, self).__init__(model)

        assert len(model.inputs) == 1, "Expected 1 input blob"
        assert len(model.outputs) == 1, "Expected 1 output blob"
        self.input_blob = next(iter(model.inputs))
        self.output_blob = next(iter(model.outputs))
        self.input_shape = model.inputs[self.input_blob].shape

        assert np.array_equal([1, self.POINTS_NUMBER * 2, 1, 1],
                model.outputs[self.output_blob].shape), \
            "Expected model output shape %s, but got %s" % \
            ([1, self.POINTS_NUMBER * 2, 1, 1],
             model.outputs[self.output_blob].shape)

    def preprocess(self, frame, rois):
        assert len(frame.shape) == 4, "Frame shape should be [1, c, h, w]"
        inputs = cut_rois(frame, rois)
        inputs = [resize_input(input, self.input_shape) for input in inputs]
        return inputs

    def enqueue(self, input):
        return super(LandmarksDetector, self).enqueue({self.input_blob: input})

    def start_async(self, frame, rois):
        inputs = self.preprocess(frame, rois)
        for input in inputs:
            self.enqueue(input)

    def get_landmarks(self):
        outputs = self.get_outputs()
        results = [ LandmarksDetector.Result(out[self.output_blob].reshape((-1, 2))) \
                    for out in outputs ]
        return results


class FacesDatabase:
    IMAGE_EXTENSIONS = ['jpg', 'png']

    class Identity:
        def __init__(self, label, descriptors):
            self.label = label
            self.descriptors = descriptors

        @staticmethod
        def cosine_dist(x, y):
            return cosine(x, y) * 0.5

    def __init__(self, path,
            face_identifier, landmarks_detector, face_detector=None):
        path = osp.abspath(path)
        self.fg_path = path
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
                rois = [ FaceDetector.Result([0, 0, 0, 0, 0, w, h]) ]

            for i, roi in enumerate(rois):
                r = [ roi ]
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
                        if name:
                            self.dump_faces(crop, descriptor, name)
                else:
                    log.debug("Adding label {} to the gallery.".format(label))
                    self.add_item(descriptor, label)

    def ask_to_save(self, image):
        save = False
        label = None
        winname = "Unknown face"
        cv2.namedWindow(winname)
        cv2.moveWindow(winname, 0, 0)
        w = int(400 * image.shape[0]/image.shape[1])
        sz = (400,w)
        resized = cv2.resize(image, sz, interpolation = cv2.INTER_AREA)
        font                   = cv2.FONT_HERSHEY_PLAIN
        bottomLeftCornerOfText = (50,50)
        fontScale              = 1
        fontColor              = (255,255,255)
        lineType               = 1
        img = cv2.copyMakeBorder(resized, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255,255,255))
        cv2.putText(img, 'This is an unrecognized image.', (30,50), font, fontScale, fontColor, lineType)
        cv2.putText(img, 'If you want to store it to the gallery,', (30,80), font, fontScale, fontColor, lineType)
        cv2.putText(img, 'please, put the name and press "Enter".', (30,110), font, fontScale, fontColor, lineType)
        cv2.putText(img, 'Otherwise, press "Escape".', (30,140), font, fontScale, fontColor, lineType)
        cv2.putText(img, 'You can see the name here:', (30,170), font, fontScale, fontColor, lineType)
        name = ''
        tmp = 0
        while(1):
            cc = img.copy()
            cv2.putText(cc, name, (30,200), font, fontScale, fontColor, lineType)
            cv2.imshow(winname, cc)

            k = cv2.waitKey(0)
            if k == 27: #Esc
                break
            if k == 13: #Enter
                if len(name) > 0:
                    save = True
                    break
                else:
                    cv2.putText(cc, "Name was not inserted. Please try again.", (30,200), font, fontScale, fontColor, lineType)
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
                matches.append( (0, 1.0) )
                continue

            id = assignments[i]
            distance = distances[i, id]
            matches.append( (id, distance) )
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


class FaceIdentifier(Module):
    # Taken from the description of the model:
    # intel_models/face-reidentification-retail-0095
    REFERENCE_LANDMARKS = [
        (30.2946 / 96, 51.6963 / 112), # left eye
        (65.5318 / 96, 51.5014 / 112), # right eye
        (48.0252 / 96, 71.7366 / 112), # nose tip
        (33.5493 / 96, 92.3655 / 112), # left lip corner
        (62.7299 / 96, 92.2041 / 112)] # right lip corner

    UNKNOWN_ID = -1
    UNKNOWN_ID_LABEL = "Unknown"

    class Result:
        def __init__(self, id, distance, desc):
            self.id = id
            self.distance = distance
            self.descriptor = desc

    def __init__(self, model, match_threshold=0.5):
        super(FaceIdentifier, self).__init__(model)

        assert len(model.inputs) == 1, "Expected 1 input blob"
        assert len(model.outputs) == 1, "Expected 1 output blob"

        self.input_blob = next(iter(model.inputs))
        self.output_blob = next(iter(model.outputs))
        self.input_shape = model.inputs[self.input_blob].shape

        assert len(model.outputs[self.output_blob].shape) == 4 or \
            len(model.outputs[self.output_blob].shape) == 2, \
            "Expected model output shape [1, n, 1, 1] or [1, n], got %s" % \
            (model.outputs[self.output_blob].shape)

        self.faces_database = None

        self.match_threshold = match_threshold

    def set_faces_database(self, database):
        self.faces_database = database

    def get_identity_label(self, id):
        if not self.faces_database or id == self.UNKNOWN_ID:
            return self.UNKNOWN_ID_LABEL
        return self.faces_database[id].label

    def preprocess(self, frame, rois, landmarks):
        assert len(frame.shape) == 4, "Frame shape should be [1, c, h, w]"
        inputs = cut_rois(frame, rois)
        self._align_rois(inputs, landmarks)
        inputs = [resize_input(input, self.input_shape) for input in inputs]
        return inputs

    def enqueue(self, input):
        return super(FaceIdentifier, self).enqueue({self.input_blob: input})

    def start_async(self, frame, rois, landmarks):
        inputs = self.preprocess(frame, rois, landmarks)
        for input in inputs:
            self.enqueue(input)

    def get_threshold(self):
        return self.match_threshold

    def get_matches(self):
        descriptors = self.get_descriptors()

        matches = []
        if len(descriptors) != 0:
            matches = self.faces_database.match_faces(descriptors)

        results = []
        unknowns_list = []
        for num, match in enumerate(matches):
            id = match[0]
            distance = match[1]
            if self.match_threshold < distance:
                id = self.UNKNOWN_ID
                unknowns_list.append(num)

            results.append(self.Result(id, distance, descriptors[num]))
        return results, unknowns_list

    def get_descriptors(self):
        return [out[self.output_blob].flatten() for out in self.get_outputs()]

    @staticmethod
    def normalize(array, axis):
        mean = array.mean(axis=axis)
        array -= mean
        std = array.std()
        array /= std
        return mean, std

    @staticmethod
    def get_transform(src, dst):
        assert np.array_equal(src.shape, dst.shape) and len(src.shape) == 2, \
            "2d input arrays are expected, got %s" % (src.shape)
        src_col_mean, src_col_std = FaceIdentifier.normalize(src, axis=(0))
        dst_col_mean, dst_col_std = FaceIdentifier.normalize(dst, axis=(0))

        u, _, vt = np.linalg.svd(np.matmul(src.T, dst))
        r = np.matmul(u, vt).T

        transform = np.empty((2, 3))
        transform[:, 0:2] = r * (dst_col_std / src_col_std)
        transform[:,   2] = dst_col_mean.T - \
            np.matmul(transform[:, 0:2], src_col_mean.T)
        return transform

    def _align_rois(self, face_images, face_landmarks):
        assert len(face_images) == len(face_landmarks), \
            "Input lengths differ, got %s and %s" % \
            (len(face_images), len(face_landmarks))

        for image, image_landmarks in zip(face_images, face_landmarks):
            assert len(image.shape) == 4, "Face image is expected"
            image = image[0]

            scale = np.array((image.shape[-1], image.shape[-2]))
            desired_landmarks = np.array(self.REFERENCE_LANDMARKS, dtype=np.float64) * scale
            landmarks = image_landmarks.get_array() * scale

            transform = FaceIdentifier.get_transform(desired_landmarks, landmarks)
            img = image.transpose((1, 2, 0))
            cv2.warpAffine(img, transform, tuple(scale), img,
                flags=cv2.WARP_INVERSE_MAP)
            image[:] = img.transpose((2, 0, 1))

class FrameProcessor:
    QUEUE_SIZE = 16

    def __init__(self, args):
        used_devices = set([args.d_fd, args.d_lm, args.d_reid])
        self.context = InferenceContext()
        context = self.context
        context.load_plugins(used_devices, args.cpu_lib, args.gpu_lib)
        for d in used_devices:
            context.get_plugin(d).set_config({
                "PERF_COUNT": "YES" if args.perf_stats else "NO"})

        log.info("Loading models")
        face_detector_net = self.load_model(args.m_fd)
        landmarks_net = self.load_model(args.m_lm)
        face_reid_net = self.load_model(args.m_reid)

        self.face_detector = FaceDetector(face_detector_net,
            confidence_threshold=args.t_fd, roi_scale_factor=args.exp_r_fd)
        self.landmarks_detector = LandmarksDetector(landmarks_net)
        self.face_identifier = FaceIdentifier(face_reid_net,
            match_threshold=args.t_id)

        self.face_detector.deploy(args.d_fd, context)
        self.landmarks_detector.deploy(args.d_lm, context,
            queue_size=self.QUEUE_SIZE)
        self.face_identifier.deploy(args.d_reid, context,
            queue_size=self.QUEUE_SIZE)
        log.info("Models are loaded")

        log.info("Building faces database using images from '%s'" % (args.fg))
        self.faces_database = FacesDatabase(args.fg, self.face_identifier,
            self.landmarks_detector,
            self.face_detector if args.run_detector else None)
        self.face_identifier.set_faces_database(self.faces_database)
        log.info("Database is built, registered %s identities" % \
            (len(self.faces_database)))

        self.allow_grow = args.allow_grow

    def load_model(self, model_path):
        model_path = osp.abspath(model_path)
        model_description_path = model_path
        model_weights_path = osp.splitext(model_path)[0] + ".bin"
        log.info("Loading the model from '%s'" % (model_description_path))
        assert osp.isfile(model_description_path), \
            "Model description is not found at '%s'" % (model_description_path)
        assert osp.isfile(model_weights_path), \
            "Model weights are not found at '%s'" % (model_weights_path)
        model = IENetwork(model_description_path, model_weights_path)
        log.info("Model is loaded")
        return model

    def process(self, frame):
        assert len(frame.shape) == 3, \
            "Expected input frame in (H, W, C) format"
        assert frame.shape[2] in [3, 4], \
            "Expected BGR or BGRA input"

        orig_image = frame.copy()
        frame = frame.transpose((2, 0, 1)) # HWC to CHW
        frame = np.expand_dims(frame, axis=0)

        self.face_detector.clear()
        self.landmarks_detector.clear()
        self.face_identifier.clear()

        self.face_detector.start_async(frame)
        rois = self.face_detector.get_roi_proposals(frame)
        if self.QUEUE_SIZE < len(rois):
            log.warning("Too many faces for processing." \
                    " Will be processed only %s of %s." % \
                    (self.QUEUE_SIZE, len(rois))
                )
            rois = rois[:self.QUEUE_SIZE]
        self.landmarks_detector.start_async(frame, rois)
        landmarks = self.landmarks_detector.get_landmarks()

        self.face_identifier.start_async(frame, rois, landmarks)
        face_identities, unknowns = self.face_identifier.get_matches()
        if self.allow_grow and len(unknowns) > 0:
            for i in unknowns:
                # This check is preventing asking to save half-images in the boundary of images
                if rois[i].position[0] == 0.0 or rois[i].position[1] == 0.0 or \
                    (rois[i].position[0] + rois[i].size[0] > orig_image.shape[1]) or \
                    (rois[i].position[1] + rois[i].size[1] > orig_image.shape[0]):
                    continue
                crop = orig_image[int(rois[i].position[1]):int(rois[i].position[1]+rois[i].size[1]), int(rois[i].position[0]):int(rois[i].position[0]+rois[i].size[0])]
                name = self.faces_database.ask_to_save(crop)
                if name:
                    id = self.faces_database.dump_faces(crop, face_identities[i].descriptor, name)
                    face_identities[i].id = id

        outputs = [rois, landmarks, face_identities]

        return outputs


    def get_performance_stats(self):
        stats = {
            'face_detector': self.face_detector.get_performance_stats(),
            'landmarks': self.landmarks_detector.get_performance_stats(),
            'face_identifier': self.face_identifier.get_performance_stats(),
        }
        return stats


class Visualizer:
    DEFAULT_CAMERA_FOCAL_DISTANCE = 950.0

    BREAK_KEY_LABEL = 'q'
    BREAK_KEY = ord(BREAK_KEY_LABEL)


    def __init__(self, args):
        self.frame_processor = FrameProcessor(args)
        self.display = not args.no_show
        self.print_perf_stats = args.perf_stats

        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0

        self.input_crop = None
        if args.crop_width and args.crop_height:
            self.input_crop = np.array((args.crop_width, args.crop_height))

        self.frame_timeout = 0 if args.timelapse else 1

    def update_fps(self):
        now = time.time()
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    def draw_text_with_background(self, frame, text, origin,
            font=cv2.FONT_HERSHEY_SIMPLEX, scale=1.0,
            color=(0, 0, 0), thickness=1, bgcolor=(255, 255, 255)):
        text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(frame,
            tuple((origin + (0, baseline)).astype(int)),
            tuple((origin + (text_size[0], -text_size[1])).astype(int)),
            bgcolor, cv2.FILLED)
        cv2.putText(frame, text,
            tuple(origin.astype(int)),
            font, scale, color, thickness)
        return text_size, baseline

    def draw_detection_roi(self, frame, roi, identity):
        label = self.frame_processor \
            .face_identifier.get_identity_label(identity.id)

        # Draw face ROI border
        cv2.rectangle(frame,
            tuple(roi.position), tuple(roi.position + roi.size),
            (0, 220, 0), 2)

        # Draw identity label
        text_scale = 0.5
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize("H1", font, text_scale, 1)
        line_height = np.array([0, text_size[0][1]])
        text = label
        if identity.id != FaceIdentifier.UNKNOWN_ID:
            text += ' %.2f%%' % (100.0 * (1 - identity.distance))
        self.draw_text_with_background(frame, text,
            roi.position - line_height * 0.5,
            font, scale=text_scale)

    def draw_detection_keypoints(self, frame, roi, landmarks):
        keypoints = [ landmarks.left_eye,
                      landmarks.right_eye,
                      landmarks.nose_tip,
                      landmarks.left_lip_corner,
                      landmarks.right_lip_corner,
                    ]

        for point in keypoints:
            center = roi.position + roi.size * point
            cv2.circle(frame, tuple(center.astype(int)), 2, (0, 255, 255), 2)

    def draw_detections(self, frame, detections):
        for roi, landmarks, identity in zip(*detections):
            self.draw_detection_roi(frame, roi, identity)
            self.draw_detection_keypoints(frame, roi, landmarks)

    def draw_status(self, frame, detections):
        origin = np.array([10, 10])
        color = (10, 160, 10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.5
        text_size, _ = self.draw_text_with_background(frame,
            "Frame time: %.3fs" % (self.frame_time),
            origin, font, text_scale, color)
        self.draw_text_with_background(frame,
            "FPS: %.1f" % (self.fps),
            (origin + (0, text_size[1] * 1.5)), font, text_scale, color)

        log.debug('Frame: %s/%s, detections: %s, ' \
                'frame time: %.3fs, fps: %.1f' % \
            (int(self.input_stream.get(cv2.CAP_PROP_POS_FRAMES)),
                int(self.input_stream.get(cv2.CAP_PROP_FRAME_COUNT)),
                len(detections[-1]), self.frame_time, self.fps
            ))

        if self.print_perf_stats:
            log.info('Performance stats:')
            log.info(self.frame_processor.get_performance_stats())

    def display_interactive_window(self, frame):
        color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.5
        text = "Press '%s' key to exit" % (self.BREAK_KEY_LABEL)
        thickness = 2
        text_size = cv2.getTextSize(text, font, text_scale, thickness)
        origin = np.array([frame.shape[-2] - text_size[0][0] - 10, 10])
        line_height = np.array([0, text_size[0][1]]) * 1.5
        cv2.putText(frame, text,
            tuple(origin.astype(int)), font, text_scale, color, thickness)

        cv2.imshow('Face recognition demo', frame)

    def should_stop_display(self):
        return cv2.waitKey(self.frame_timeout) & 0xFF == self.BREAK_KEY

    def process(self, input_stream, output_stream):
        self.input_stream = input_stream
        self.output_stream = output_stream

        while input_stream.isOpened():
            has_frame, frame = input_stream.read()
            if not has_frame:
                break

            if self.input_crop is not None:
                frame = Visualizer.center_crop(frame, self.input_crop)
            detections = self.frame_processor.process(frame)

            self.draw_detections(frame, detections)
            self.draw_status(frame, detections)

            if output_stream:
                output_stream.write(frame)
            if self.display:
                self.display_interactive_window(frame)
                if self.should_stop_display():
                    break

            self.update_fps()

    @staticmethod
    def center_crop(frame, crop_size):
        fh, fw, fc = frame.shape
        crop_size[0] = min(fw, crop_size[0])
        crop_size[1] = min(fh, crop_size[1])
        return frame[(fh - crop_size[1]) // 2 : (fh + crop_size[1]) // 2,
                     (fw - crop_size[0]) // 2 : (fw + crop_size[0]) // 2,
                     :]

    def run(self, args):
        input_stream = Visualizer.open_input_stream(args.input)
        fps = input_stream.get(cv2.CAP_PROP_FPS)
        frame_size = (int(input_stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(input_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if args.crop_width and args.crop_height:
            crop_size = (args.crop_width, args.crop_height)
            frame_size = tuple(np.minimum(frame_size, crop_size))
        log.info("Input stream info: %d x %d @ %.2f FPS" % \
            (frame_size[0], frame_size[1], fps))
        output_stream = Visualizer.open_output_stream(args.output, fps, frame_size)

        self.process(input_stream, output_stream)

        # Release resources
        if output_stream:
            output_stream.release()
        if input_stream:
            input_stream.release()

        cv2.destroyAllWindows()

    @staticmethod
    def open_input_stream(path):
        log.info("Reading input data from '%s'" % (path))
        if path == 'cam':
            stream = 0
        else:
            assert osp.isfile(path), "Input file '%s' not found" % (path)
            stream = path
        return cv2.VideoCapture(stream)

    @staticmethod
    def open_output_stream(path, fps, frame_size):
        output_stream = None
        if path != "":
            if not path.endswith('.avi'):
                log.warning("Output file extension is not 'avi'. " \
                        "Some issues with output can occur, check logs.")
            log.info("Writing output to '%s'" % (path))
            output_stream = cv2.VideoWriter(path,
                cv2.VideoWriter.fourcc(*'MJPG'), fps, frame_size)
        return output_stream


def main():
    args = build_argparser().parse_args()

    log.basicConfig(format="[ %(levelname)s ] %(asctime)-15s %(message)s",
        level=log.INFO if not args.verbose else log.DEBUG, stream=sys.stdout)

    log.debug(str(args))

    visualizer = Visualizer(args)
    visualizer.run(args)


if __name__ == '__main__':
    main()
