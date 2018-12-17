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

from __future__ import print_function
import sys
import os
import os.path as osp
from argparse import ArgumentParser
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import math
import time
import logging as log
from openvino.inference_engine import IENetwork, IEPlugin

DEVICE_KINDS = ['CPU', 'GPU', 'FPGA', 'MYRIAD', 'HETERO']

def build_argparser():
    parser = ArgumentParser()

    general = parser.add_argument_group('General')
    general.add_argument('-i', '--input', metavar="PATH", default='cam',
        help="Path to the input video ('cam' for the camera, default)")
    general.add_argument('-o', '--output', metavar="PATH", default="",
        help="(optional) Path to save the output video to")
    general.add_argument('-no_show', action='store_true',
        help="(optional) Do not display output")
    general.add_argument('-cw', '--crop_width', default=0, type=int,
        help="Crop input stream to this width.")
    general.add_argument('-ch', '--crop_height', default=0, type=int,
        help="Crop input stream to this height.")

    faces = parser.add_argument_group('Faces database')
    faces.add_argument('-fg', metavar="PATH", required=True,
        help="Path to the face images directory")

    models = parser.add_argument_group('Models')
    models.add_argument('-m_fd', metavar="PATH", default="", required=True,
        help="Path to the " \
            "Face Detection Retail model XML file")
    models.add_argument('-m_lm', metavar="PATH", default="",
        help="(optional) Path to the " \
            "Facial Landmarks Regression Retail model XML file")
    models.add_argument('-m_reid', metavar="PATH", default="",
        help="(optional) Path to the " \
            "Face Reidentification Retail model XML file")
    models.add_argument('-m_hp', metavar="PATH", default="",
        help="(optional) Path to the " \
            "Head Pose Estimation Retail model XML file")

    infer = parser.add_argument_group('Inference options')
    infer.add_argument('-d_fd', default='CPU', choices=DEVICE_KINDS,
        help="(optional) Target device for the " \
            "Face Detection Retail model " \
            "(default: %(default)s)")
    infer.add_argument('-d_lm', default='CPU', choices=DEVICE_KINDS,
        help="(optional) Target device for the " \
            "Facial Landmarks Regression Retail model " \
            "(default: %(default)s)")
    infer.add_argument('-d_reid', default='CPU', choices=DEVICE_KINDS,
        help="(optional) Target device for the " \
            "Face Reidentification Retail model " \
            "(default: %(default)s)")
    infer.add_argument('-d_hp', default='CPU', choices=DEVICE_KINDS,
        help="(optional) Target device for the " \
            "Head Pose Estimation Retail model " \
            "(default: %(default)s)")
    infer.add_argument('-l', '--cpu_lib', metavar="PATH", default="",
        help="(optional) For MKLDNN (CPU)-targeted custom layers, " \
            "if any. Path to a shared library with " \
            "implementations of the kernels")
    # infer.add_argument('-c', '--gpu_lib', metavar="PATH", default="",
    #     help="(optional) For clDNN (GPU)-targeted custom layers, " \
    #         "if any. Path to the XML file with " \
    #         "descriptions of the kernels")
    # infer.add_argument('-pc', action='store_true',
    #     help="(optional) Output per-layer performance statistics")
    infer.add_argument('--verbose', action='store_true',
        help="(optional) Be more verbose")
    infer.add_argument('-t_fd', metavar='[0..1]', type=float, default=0.6,
        help="(optional) Probability threshold for face detections" \
            "(default: %(default)s)")
    infer.add_argument('-t_id', metavar='[0..1]', type=float, default=0.3,
        help="(optional) Cosine distance threshold between two vectors " \
            "for face identification" \
            "(default: %(default)s)")
    infer.add_argument('-exp_r_fd', metavar='NUMBER', type=float, default=1.15,
        help="(optional) Scaling ratio for bbox passed to face recognition" \
            "(default: %(default)s)")

    return parser

class InferenceContext:
    def __init__(self):
        self.plugins = {}

    def load_plugins(self, devices, cpu_ext=""):
        log.info("Loading plugins for devices: %s" % (devices))

        plugins = { d: IEPlugin(d) for d in devices }
        if 'CPU' in plugins and not len(cpu_ext) == 0:
            log.info("Using CPU extensions library '%s'" % (cpu_ext))
            assert osp.isfile(cpu_ext), "Failed to open CPU extensions library"
            plugins['CPU'].add_cpu_extension(cpu_ext)
        self.plugins = plugins

        log.info("Plugins are loaded")

    def check_model_support(self, net, device):
        plugin = self.plugins[device]

        if plugin.device == "CPU":
            supported_layers = plugin.get_supported_layers(net)
            not_supported_layers = [l for l in net.layers.keys() \
                                    if l not in supported_layers]
            if len(not_supported_layers) != 0:
                log.error("The following layers are not supported " \
                    "by the plugin for the specified device {}:\n {}". \
                    format(plugin.device, ', '.join(not_supported_layers)))
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
    p1 = np.maximum(np.minimum(p1, frame.shape[-2:]), [0, 0])
    p2 = (roi.position + roi.size).astype(int)
    p2 = np.maximum(np.minimum(p2, frame.shape[-2:]), [0, 0])
    return np.array(frame[ :, :, p1[1]:p2[1], p1[0]:p2[0] ])

def cut_rois(frame, rois):
    return [cut_roi(frame, roi) for roi in rois]

class Module:
    def __init__(self, model):
        self.model = model
        self.device_model = None

        self.max_requests = 0
        self.active_requests = 0

        self.outputs = []

    def deploy(self, device, context, queue_size=1):
        self.context = context
        self.max_requests = queue_size
        self.device_model = context.deploy_model(
            self.model, device, self.max_requests)
        self.model = None

    def enqueue(self, input):
        self.outputs = []

        if self.max_requests <= self.active_requests:
            log.warn("Processing request rejected - too much requests")
            return False

        self.device_model.start_async(self.active_requests, input)
        self.active_requests += 1
        return True

    def wait(self):
        if self.active_requests <= 0:
            return

        self.outputs = [None, ] * self.active_requests
        for i in range(self.active_requests):
            self.device_model.requests[i].wait()
            self.outputs[i] = self.device_model.requests[i].outputs

        self.active_requests = 0

    def get_outputs(self):
        self.wait()
        outputs = self.outputs
        self.outputs = []
        return outputs

    def adjust_size(self, frame, target_shape):
        assert len(frame.shape) == len(target_shape), \
            "Expected a frame with %s dimensions, but got %s" % \
            (len(target_shape), len(frame.shape))

        assert frame.shape[0] == 1, "Only batch size of 1 is supported"
        n, c, h, w = target_shape
        
        input = frame[0]
        if True or tuple(target_shape[-2:]) != tuple(frame.shape[-2:]):
            input = input.transpose((1, 2, 0)) # to HWC
            input = cv2.resize(input, (w, h))
            input = input.transpose((2, 0, 1)) # to CHW

        return input.reshape((n, c, h, w))

class FaceDetector(Module):
    class Result:
        OUTPUT_SIZE = 7

        def __init__(self, output):
            self.image_id = output[0]
            self.label = int(output[1])
            self.confidence = output[2]
            self.position = np.array((output[3], output[4])) # x, y
            self.size = np.array((output[5], output[6])) # width, height

        def __str__(self):
            return str(vars(self))

        def __repr__(self):
            return str(vars(self))

    def __init__(self, model, confidence_threshold=0.5, roi_scale_factor=1.15):
        super(FaceDetector, self).__init__(model)

        assert len(model.inputs) == 1, \
            "The model is expected to have 1 input blob"
        assert len(model.outputs) == 1, \
            "The model is expected to have 1 output blob"
        self.input_blob = next(iter(model.inputs))
        self.output_blob = next(iter(model.outputs))
        self.input_shape = model.inputs[self.input_blob].shape
        self.output_shape = model.outputs[self.output_blob].shape

        assert tuple(self.input_shape) == (1, 3, 300, 300), \
            "The model is expected to have an input shape %s, but has %s" % \
            ((1, 3, 300, 300), self.input_shape)

        assert len(self.output_shape) == 4 and \
            self.output_shape[3] == self.Result.OUTPUT_SIZE, \
            "Model is expected to have %s outputs" % (self.Result.OUTPUT_SIZE)

        assert 0.0 <= confidence_threshold and confidence_threshold <= 1.0, \
            "Confidence threshold is expected to be in range [0; 1]"
        self.confidence_threshold = confidence_threshold

        assert 0.0 < roi_scale_factor, "Expected positive ROI scale factor"
        self.roi_scale_factor = roi_scale_factor

    def get_input_shape(self):
        return self.input_shape

    def preprocess(self, frame):
        # inputs = [ frame ]
        inputs = [ self.adjust_size(frame, self.input_shape) ]
        return inputs

    def start_async(self, frame):
        inputs = self.preprocess(frame)
        for input in inputs:
            self.enqueue(input)

    def enqueue(self, input):
        # Expected input: (1, 3, 300, 300) BGR
        return super(FaceDetector, self).enqueue({self.input_blob: input})

    def get_roi_proposals(self, frame):
        self.start_async(frame)
        outputs = self.get_outputs()[0][self.output_blob]
        # outputs shape is [N_requests, 1, 1, N_found_faces, 7]

        frame_width = frame.shape[-1]
        frame_height = frame.shape[-2]

        results = []
        for output in outputs[0][0]:
            result = self.Result(output)
            if result.confidence < self.confidence_threshold:
                continue

            self._rescale(result, frame_width, frame_height)
            self._centrify_squarify(result, self.roi_scale_factor)
            results.append(result)

        return results

    def _centrify_squarify(self, result, roi_scale_factor=1.0):
        bbox = result.size
        bbox_center = result.position + result.size * 0.5
        max_side = max(bbox)
        bbox_square = np.repeat(roi_scale_factor * max_side, 2)
        result.position[:] = bbox_center - bbox_square * 0.5
        result.size[:] = bbox_square

        return result

    def _rescale(self, result, frame_width, frame_height):
        result.position[0] = result.position[0] * frame_width
        result.position[1] = result.position[1] * frame_height
        result.size[0] = result.size[0] * frame_width - result.position[0]
        result.size[1] = result.size[1] * frame_height - result.position[1]
        return result

class HeadPoseEstimator(Module):
    OUTPUT_PITCH = 'angle_p_fc'
    OUTPUT_YAW = 'angle_y_fc'
    OUTPUT_ROLL = 'angle_r_fc'

    class Result:
        def __init__(self, pitch, yaw, roll):
            self.pitch = pitch
            self.yaw = yaw
            self.roll = roll

        def __str__(self):
            return str(vars(self))

        def __repr__(self):
            return str(vars(self))

    def __init__(self, model):
        super(HeadPoseEstimator, self).__init__(model)

        assert len(model.inputs) == 1, \
            "The model is expected to have 1 input blob"
        assert len(model.outputs) == 3, \
            "The model is expected to have 3 output blobs"
        self.input_blob = next(iter(model.inputs))
        self.input_shape = model.inputs[self.input_blob].shape

        assert tuple(self.input_shape) == (1, 3, 60, 60), \
            "The model is expected to have an input shape %s, but has %s" % \
            ((1, 3, 60, 60), self.input_shape)

        assert model.outputs[self.OUTPUT_PITCH].shape == [1, 1], \
            "Expected '%s' blob output shape (1, 1), got %s" % \
            (self.OUTPUT_PITCH, model.outputs[self.OUTPUT_PITCH].shape)
        assert model.outputs[self.OUTPUT_YAW].shape == [1, 1], \
            "Expected '%s' blob output shape (1, 1), got %s" % \
            (self.OUTPUT_YAW, model.outputs[self.OUTPUT_YAW].shape)
        assert model.outputs[self.OUTPUT_ROLL].shape == [1, 1], \
            "Expected '%s' blob output shape (1, 1), got %s" % \
            (self.OUTPUT_ROLL, model.outputs[self.OUTPUT_ROLL].shape)

    def preprocess(self, frame, rois):
        inputs = cut_rois(frame, rois)
        inputs = [self.adjust_size(input, self.input_shape) for input in inputs]
        return inputs

    def enqueue(self, input):
        # Expected input: (1, 3, 60, 60) BGR
        return super(HeadPoseEstimator, self).enqueue({self.input_blob: input})

    def start_async(self, frame, rois):
        inputs = self.preprocess(frame, rois)
        for input in inputs:
            self.enqueue(input)

    def get_head_poses(self):
        outputs = self.get_outputs()

        results = []
        for output in outputs:
            pitch = output[self.OUTPUT_PITCH][0][0]
            yaw = output[self.OUTPUT_YAW][0][0]
            roll = output[self.OUTPUT_ROLL][0][0]
            results.append(self.Result(pitch, yaw, roll))

        return results

class LandmarksDetector(Module):
    # Taken from:
    # https://github.com/opencv/open_model_zoo/blob/2018/intel_models/facial-landmarks-35-adas-0001/description/facial-landmarks-35-adas-0001.md
    DEFAULT_CHANNEL_MEAN = np.array((120, 110, 104), dtype=np.uint8)

    POINTS_NUMBER = 35

    class Result:
        def __init__(self, outputs):
            self.points = outputs[0]

            p = lambda i: self[i]
            self.left_eye = (p(0) + p(1)) * 0.5
            self.right_eye = (p(2) + p(3)) * 0.5
            self.nose_tip = p(4)
            self.left_lip_corner = p(8)
            self.right_lip_corner = p(9)

        def __str__(self):
            return str(vars(self))

        def __repr__(self):
            return str(vars(self))

        def __getitem__(self, idx):
            return np.array(self.points[idx : idx + 2])

    def __init__(self, model):
        super(LandmarksDetector, self).__init__(model)

        assert len(model.inputs) == 1, \
            "The model is expected to have 1 input blob"
        assert len(model.outputs) == 1, \
            "The model is expected to have 1 output blob"
        self.input_blob = next(iter(model.inputs))
        self.output_blob = next(iter(model.outputs))
        self.input_shape = model.inputs[self.input_blob].shape

        assert tuple(self.input_shape) == (1, 3, 60, 60), \
            "The model is expected to have an input shape %s, but has %s" % \
            ((1, 3, 60, 60), self.input_shape)

        assert model.outputs[self.output_blob].shape == \
               [1, self.POINTS_NUMBER * 2], \
            "Expected output shape (1, %s), but got %s" % \
            (self.POINTS_NUMBER * 2, model.outputs[self.output_blob].shape)

    def preprocess(self, frame, rois):
        for channel, mean in zip(frame[0], self.DEFAULT_CHANNEL_MEAN):
            channel -= mean

        inputs = cut_rois(frame, rois)
        inputs = [self.adjust_size(input, self.input_shape) for input in inputs]
        return inputs

    def enqueue(self, input):
        # Expected input: (1, 3, 60, 60) BGR mean subtracted
        return super(LandmarksDetector, self).enqueue({self.input_blob: input})

    def start_async(self, frame, rois):
        inputs = self.preprocess(frame, rois)
        for input in inputs:
            self.enqueue(input)

    def get_landmarks(self):
        outputs = self.get_outputs()

        results = []
        for output in outputs:
            results.append(self.Result(output[self.output_blob]))

        return results

class FacesDatabase:
    IMAGE_EXTENSIONS = [ '.jpg', '.png' ]

    class Identity:
        def __init__(self, label, descriptor):
            self.label = label
            self.descriptor = descriptor

    def __init__(self, path, face_identifier_model):
        path = osp.abspath(path)
        paths = []
        if osp.isdir(path):
            ext = self.IMAGE_EXTENSIONS
            paths = [osp.join(path, f) for f in os.listdir(path) \
                      if f.endswith(ext[0]) or f.endswith(ext[1])]
        else:
            raise Exception("Wrong face images database path. Expected path to" \
                            "to a directory containing '%s' files, got '%s'" % \
                            (self.IMAGE_EXTENSIONS, path))

        self.database = []
        for path in paths:
            label = osp.splitext(osp.basename(path))[0]
            image = cv2.imread(path)

            n, c, h, w = face_identifier_model.get_input_shape()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (w, h))
            image = image.transpose((2, 0, 1)) # HWC to CHW
            image = image.reshape((n, c, h, w))

            descriptor = face_identifier_model.get_descriptors([image])[0]
            self.database.append(self.Identity(label, descriptor))

    def match_faces(self, descriptors):
        database = self
        distances = np.empty((len(descriptors), len(database)))
        for i, desc in enumerate(descriptors):
            for j, identity in enumerate(database):
                distances[i][j] = self._cosine_distance(desc, identity.descriptor)

        # Find best assignments, prevent repeats (assuming faces can not repeat)
        _, assignments = linear_sum_assignment(distances)
        matches = []
        for i in range(len(descriptors)):
            id = assignments[i]
            distance = distances[i, id]
            matches.append( (id, distance) )
        return matches

    def __getitem__(self, idx):
        return self.database[idx]

    def __len__(self):
        return len(self.database)

    def _cosine_distance(self, x, y):
        norm = math.sqrt(np.dot(x, x) * np.dot(y, y)) + 1e-6
        return 1.0 - np.dot(x, y) / norm

class FaceIdentifier(Module):
    # Taken from:
    # https://github.com/opencv/open_model_zoo/blob/2018/intel_models/face-reidentification-retail-0071/description/face-reidentification-retail-0071.md
    REFERENCE_LANDMARKS = {
        "left_eye": (0.31556875, 0.46157410),
        "right_eye": (0.68262292, 0.46157410),
        "nose_tip": (0.50026249, 0.64050535),
        "left_lip_corner": (0.34947187, 0.82469196),
        "right_lip_corner": (0.65343645, 0.82469196)
    }

    UNKNOWN_ID = -1
    UNKNOWN_ID_LABEL = "Unknown"

    class Result:
        def __init__(self, id, distance):
            self.id = id
            self.distance = distance

        def __str__(self):
            return str(vars(self))

        def __repr__(self):
            return str(vars(self))

    def __init__(self, model, match_threshold=0.5):
        super(FaceIdentifier, self).__init__(model)

        assert len(model.inputs) == 1, \
            "The model is expected to have 1 input blob"
        assert len(model.outputs) == 1, \
            "The model is expected to have 1 output blob"

        self.input_blob = next(iter(model.inputs))
        self.output_blob = next(iter(model.outputs))
        self.input_shape = model.inputs[self.input_blob].shape

        assert tuple(self.input_shape) == (1, 3, 128, 128), \
            "The model is expected to have an input shape %s, but has %s" % \
            ((1, 3, 128, 128), self.input_shape)

        assert len(model.outputs[self.output_blob].shape) == 4, \
            "Expected output shape (1, n, 1, 1), got %s" % \
            (model.outputs[self.output_blob].shape)

        self.faces_database = None

        self.match_threshold = match_threshold

    def get_input_shape(self):
        return self.input_shape

    def build_faces_database(self, database_path):
        log.info("Building faces database using images from '%s'" % \
            (database_path))

        self.faces_database = FacesDatabase(database_path, self)

        log.info("Database is built, registered %s identities" % \
            (len(self.faces_database)))

    def get_identity_label(self, id):
        if not self.faces_database or id == self.UNKNOWN_ID:
            return self.UNKNOWN_ID_LABEL
        return self.faces_database[id].label

    def preprocess(self, frame, rois, landmarks):
        assert len(frame.shape) == 4, "Frame shape should be (1, c, h, w)"
        frame = self._adjust_color(frame)
        inputs = cut_rois(frame, rois)
        self._align_rois(inputs, landmarks)
        inputs = [self.adjust_size(input, self.input_shape) for input in inputs]
        return inputs

    def enqueue(self, input):
        # Expected input: (1, 3, 128, 128) RGB
        return super(FaceIdentifier, self).enqueue({self.input_blob: input})

    def start_async(self, frame, rois, landmarks):
        inputs = self.preprocess(frame, rois, landmarks)
        for input in inputs:
            self.enqueue(input)

    def get_matches(self):
        outputs = self.get_outputs()
        descriptors = [out[self.output_blob].squeeze() for out in outputs]

        matches = []
        if len(descriptors) != 0:
            matches = self.faces_database.match_faces(descriptors)

        results = []
        for match in matches:
            id = match[0]
            distance = match[1]
            if self.match_threshold < distance:
                id = self.UNKNOWN_ID
            results.append(self.Result(id, distance))
        return results

    def get_descriptors(self, face_images):
        results = []
        for image in face_images:
            self.enqueue(image)
            results.append(self.get_outputs()[0][self.output_blob].squeeze())
        return results

    def _normalize(self, array, axis):
        mean = array.mean(axis=axis)
        std = array.std(axis=axis)
        array[:] = (array - mean) / std
        return mean, std

    def _get_transform(self, src, dst):
        assert np.all(src.shape == dst.shape) and len(src.shape) == 2, \
            "2d input arrays are expected, got %s" % (src.shape)
        src = np.matrix(src)
        dst = np.matrix(dst)
        src_col_mean, src_col_std = self._normalize(src, axis=(0))
        dst_col_mean, dst_col_std = self._normalize(dst, axis=(0))

        s, u, vt = np.linalg.svd(src.T * dst)
        r = (u * vt).T

        transform = np.matrix(np.empty((2, 3)))
        transform[:, 0:2] = r * (dst_col_std / src_col_std)
        transform[:,   2] = dst_col_std.T - \
            transform[:, 0:2] * src_col_mean.T
        return transform

    def _align_rois(self, face_images, face_landmarks):
        assert len(face_images) == len(face_landmarks), \
            "Input lengths differ, got %s and %s" % \
            (len(face_images), len(face_landmarks))

        for image, image_landmarks in zip(face_images, face_landmarks):
            assert len(image.shape) == 4, "Face image is expected"
            image = image[0]

            scale = np.array(image.shape[1:3])
            desired_landmarks = np.array([
                self.REFERENCE_LANDMARKS["left_eye"],
                self.REFERENCE_LANDMARKS["right_eye"],
                self.REFERENCE_LANDMARKS["nose_tip"],
                self.REFERENCE_LANDMARKS["left_lip_corner"],
                self.REFERENCE_LANDMARKS["right_lip_corner"],
            ]) * scale

            landmarks = np.array([
                image_landmarks.left_eye,
                image_landmarks.right_eye,
                image_landmarks.nose_tip,
                image_landmarks.left_lip_corner,
                image_landmarks.right_lip_corner,
            ]) * scale

            transform = self._get_transform(desired_landmarks, landmarks)
            cv2.warpAffine(image, transform, tuple(scale), image,
                cv2.WARP_INVERSE_MAP)

    def _adjust_color(self, frame):
        assert len(frame.shape) == 4 and frame.shape[1] == 3, \
            "Expected input of shape (1, 3, h, w) in BGR format"
        # convert BGR to RGB
        image = frame[0]
        return np.array([[image[2], image[1], image[0]]])

class FrameProcessor:
    QUEUE_SIZE = 16

    def __init__(self, args):
        used_devices = set([args.d_fd, args.d_hp, args.d_lm, args.d_reid])
        self.context = InferenceContext()
        context = self.context
        context.load_plugins(used_devices, args.cpu_lib)

        log.info("Loading models")
        face_detector_net = self.load_model(args.m_fd)
        head_pose_net = self.load_model(args.m_hp)
        landmarks_net = self.load_model(args.m_lm)
        face_reid_net = self.load_model(args.m_reid)

        self.face_detector = FaceDetector(face_detector_net,
            confidence_threshold=args.t_fd, roi_scale_factor=args.exp_r_fd)
        self.head_pose_estimator = HeadPoseEstimator(head_pose_net)
        self.landmarks_detector = LandmarksDetector(landmarks_net)
        self.face_identifier = FaceIdentifier(face_reid_net,
            match_threshold=args.t_id)

        self.face_detector.deploy(args.d_fd, context)
        self.head_pose_estimator.deploy(args.d_hp, context,
            queue_size=self.QUEUE_SIZE)
        self.landmarks_detector.deploy(args.d_lm, context,
            queue_size=self.QUEUE_SIZE)
        self.face_identifier.deploy(args.d_reid, context,
            queue_size=self.QUEUE_SIZE)
        log.info("Models are loaded")

        self.face_identifier.build_faces_database(args.fg)

    def load_model(self, model_path):
        model_path = osp.abspath(model_path)
        model_description_path = model_path
        model_weights_path = osp.splitext(model_path)[0] + ".bin"
        log.info("Loading the model from '%s'" % (model_description_path))
        assert osp.isfile(model_description_path), \
            "Model description is not found at '%s'" % (model_description_path)
        assert osp.isfile(model_weights_path), \
            "Model weights are not found at '%s'" % (model_weights_path)
        model = IENetwork.from_ir(model_description_path, model_weights_path)
        log.info("Model is loaded")
        return model

    def process(self, frame):
        input_size = (frame.shape[1], frame.shape[0])
        n, c, h, w = self.face_detector.get_input_shape()
        frame = cv2.resize(frame, (w, h))
        frame = frame.transpose((2, 0, 1)) # HWC to CHW
        frame = frame.reshape((n, c, h, w))

        rois = self.face_detector.get_roi_proposals(frame)
        if self.QUEUE_SIZE < len(rois):
            log.warn("Too much faces for processing, dropping all " \
                "the redundant. Will be processed only %s of total %s." % \
                (self.QUEUE_SIZE, len(rois))
                )
            rois = rois[:self.QUEUE_SIZE]
        self.head_pose_estimator.start_async(frame, rois)
        self.landmarks_detector.start_async(frame, rois)
        head_poses = self.head_pose_estimator.get_head_poses()
        landmarks = self.landmarks_detector.get_landmarks()

        self.face_identifier.start_async(frame, rois, landmarks)
        face_identities = self.face_identifier.get_matches()

        outputs = [rois, head_poses, landmarks, face_identities]
        outputs = self.rescale_detections(outputs, (w, h), input_size)

        return outputs

    def rescale_detections(self, detections, current_size, target_size):
        scale = np.array(target_size) / np.array(current_size)
        for roi, head_pose, landmarks, face_id in zip(*detections):
            roi.position *= scale
            roi.size *= scale
        return detections

class Visualizer:
    DEFAULT_CAMERA_FOCAL_DISTANCE = 950.0

    def __init__(self, args):
        self.frame_processor = FrameProcessor(args)
        self.display = not args.no_show
        self.verbose = args.verbose

        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0

        self.input_crop = None
        if args.crop_width and args.crop_height:
            self.input_crop = np.array((args.crop_width, args.crop_height))

    def update_fps(self):
        now = time.time()
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    def draw_detection_roi(self, frame, roi, identity):
        label = self.frame_processor \
            .face_identifier.get_identity_label(identity.id)
        color = (int(min(identity.id * 12.5, 255)),
                 int(min(identity.id * 7, 255)),
                 int(min(identity.id * 5, 255)))

        # Draw face ROI border
        cv2.rectangle(frame,
            tuple(roi.position), tuple(roi.position + roi.size), color)

        # Draw identity label
        text_scale = 0.5
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize("H1", font, text_scale, 1)
        line_height = np.array([0, text_size[0][1]])
        cv2.putText(frame, '%s %.2f%%' % (label, 100.0 * (1 - identity.distance)),
            tuple((roi.position - line_height * 0.5).astype(int)),
            font, text_scale, (200, 200, 200))

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

    def build_camera_matrix(self, center_x, center_y, focal_distance):
        camera_matrix = np.matrix(np.zeros((3, 3)))
        camera_matrix[0, 0] = focal_distance
        camera_matrix[0, 2] = center_x
        camera_matrix[1, 1] = focal_distance
        camera_matrix[1, 2] = center_y
        camera_matrix[2, 2] = 1
        return camera_matrix

    def build_rotation_matrix(self, pitch, yaw, roll):
        from math import sin, cos

        rotation_x = np.matrix([[1,                     0,            0],
                                [0,            cos(pitch),  -sin(pitch)],
                                [0,            sin(pitch),   cos(pitch)]])

        rotation_y = np.matrix([[cos(yaw),              0,    -sin(yaw)],
                                [0,                     1,            0],
                                [sin(yaw),              0,     cos(yaw)]])

        rotation_z = np.matrix([[cos(roll),    -sin(roll),            0],
                                [sin(roll),     cos(roll),            0],
                                [0,                     0,            1]])

        return rotation_z * rotation_y * rotation_x

    def draw_axes(self, frame, angles, center, focal_distance, scale=1.0):
        from math import radians
        pitch = radians(angles.pitch)
        yaw = radians(angles.yaw)
        roll = radians(angles.roll)

        r = self.build_rotation_matrix(pitch, yaw, roll)

        camera_matrix = self.build_camera_matrix(
            frame.shape[-1] / 2, frame.shape[-2] / 2, focal_distance)

        origin = np.matrix([0, 0, camera_matrix[0, 0]]).T

        x_axis =  r * np.matrix([1 * scale,          0,          0]).T + origin
        y_axis =  r * np.matrix([0,         -1 * scale,          0]).T + origin
        z_axis =  r * np.matrix([0,                  0, -1 * scale]).T + origin
        z_axis1 = r * np.matrix([0,                  0,  1 * scale]).T + origin

        def make_point(center, axis, distance):
            return center + \
                axis[0:2] / axis[2] * np.diag(camera_matrix)[0:2] * distance

        distance = 20
        cv2.line(frame,
            tuple(center.astype(int)),
            tuple(make_point(center, x_axis.A1, distance).astype(int)),
            (0, 0, 255), 2)

        cv2.line(frame,
            tuple(center.astype(int)),
            tuple(make_point(center, y_axis.A1, distance).astype(int)),
            (0, 255, 0), 2)

        p1 = make_point(center, z_axis.A1, distance).astype(int)
        p2 = make_point(center, z_axis1.A1, distance).astype(int)
        cv2.line(frame, tuple(p1), tuple(p2), (255, 0, 0), 2)
        cv2.circle(frame, tuple(p2), 2, (255, 0, 0), 2)

    def draw_detection_direction(self, frame, roi, pose):
        self.draw_axes(frame, pose, roi.position + roi.size * 0.5,
            focal_distance=self.DEFAULT_CAMERA_FOCAL_DISTANCE)

    def draw_detections(self, frame, detections):
        for roi, head_pose, landmarks, identity in zip(*detections):
            self.draw_detection_roi(frame, roi, identity)
            self.draw_detection_keypoints(frame, roi, landmarks)
            self.draw_detection_direction(frame, roi, head_pose)

    def draw_status(self, frame, detections):
        origin = np.array([10, 10])
        color = (10, 160, 10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.5
        text_size = cv2.getTextSize("H1", font, text_scale, 1)
        line_height = np.array([0, text_size[0][1]]) * 1.5
        cv2.putText(frame, "Frame time: %.3fs" % (self.frame_time),
            tuple(origin.astype(int)), font, text_scale, color)
        cv2.putText(frame, "FPS: %.1f" % (self.fps),
            tuple((origin + line_height).astype(int)), font, text_scale, color)

        if self.verbose:
            log.info('Frame: %s/%s, detections: %s, ' \
                     'frame time: %.3fs, fps: %.1f' % \
                (int(self.input_stream.get(cv2.CAP_PROP_POS_FRAMES)),
                 int(self.input_stream.get(cv2.CAP_PROP_FRAME_COUNT)),
                 len(detections[3]), self.frame_time, self.fps
                ))

    def process(self, input_stream, output_stream):
        self.input_stream = input_stream
        self.output_stream = output_stream

        has_frame, frame = input_stream.read()
        while input_stream.isOpened():
            has_frame, frame = input_stream.read()
            if not has_frame:
                break

            if self.input_crop is not None:
                frame = self.center_crop(frame, self.input_crop)
            detections = self.frame_processor.process(frame)

            self.draw_detections(frame, detections)
            self.draw_status(frame, detections)

            if self.display:
                cv2.imshow('Face recognition demo', frame)
            if output_stream:
                output_stream.write(frame)

            self.update_fps()

    def center_crop(self, frame, crop_size):
        fh, fw, fc = frame.shape
        return frame[(fh - crop_size[1]) // 2 : (fh + crop_size[1]) // 2,
                     (fw - crop_size[0]) // 2 : (fw + crop_size[0]) // 2,
                     :]

    def run(self, args):
        input_stream = open_input_stream(args.input)
        fps = input_stream.get(cv2.CAP_PROP_FPS)
        frame_size = (args.crop_width or \
                int(input_stream.get(cv2.CAP_PROP_FRAME_WIDTH)), 
            args.crop_height or \
                int(input_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
        log.info("Input stream info: %d x %d @ %.2f FPS" % \
            (frame_size[0], frame_size[1], fps))
        output_stream = open_output_stream(args.output, fps, frame_size)

        self.process(input_stream, output_stream)

        # Release resources
        if output_stream:
            output_stream.release()
        if input_stream:
            input_stream.release()

        cv2.destroyAllWindows()

def open_input_stream(path):
    log.info("Reading input data from '%s'" % (path))
    if path == 'cam':
        stream = 0
    else:
        assert osp.isfile(path), "Input file '%s' not found" % (path)
        stream = path
    return cv2.VideoCapture(stream)

def open_output_stream(path, fps, frame_size):
    output_stream = None
    if path != "":
        if not path.endswith('.avi'):
            log.warn("Output file extension is not 'avi'. " \
                     "Some issues with output can occur, check logs.")
        log.info("Writing output to '%s'" % (path))
        output_stream = cv2.VideoWriter(path,
            cv2.VideoWriter.fourcc(*'MJPG'), fps, frame_size)
    return output_stream

def main():
    log.basicConfig(format="[ %(levelname)s ] %(asctime)-15s %(message)s",
        level=log.INFO, stream=sys.stdout)

    args = build_argparser().parse_args()
    log.info(str(args))

    visualizer = Visualizer(args)
    visualizer.run(args)

if __name__ == '__main__':
    main()
