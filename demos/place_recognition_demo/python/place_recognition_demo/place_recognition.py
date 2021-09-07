"""
 Copyright (c) 2021 Intel Corporation

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

import numpy as np

import cv2
from tqdm import tqdm

from place_recognition_demo.common import crop_resize

from openvino.inference_engine import IECore # pylint: disable=no-name-in-module


class IEModel: # pylint: disable=too-few-public-methods
    """ Class that allows working with Inference Engine model. """

    def __init__(self, model_path, device, cpu_extension):
        ie = IECore()
        if cpu_extension and device == 'CPU':
            ie.add_extension(cpu_extension, 'CPU')

        self.net = ie.read_network(model_path, model_path.with_suffix('.bin'))
        self.input_name = next(iter(self.net.input_info))
        self.output_name = next(iter(self.net.outputs))
        self.input_size = self.net.input_info[self.input_name].input_data.shape
        self.exec_net = ie.load_network(network=self.net, device_name=device)

    def predict(self, image):
        ''' Takes input image and returns L2-normalized embedding vector. '''

        assert len(image.shape) == 4
        image = np.transpose(image, (0, 3, 1, 2))
        out = self.exec_net.infer(inputs={self.input_name: image})[self.output_name]
        return out


class PlaceRecognition:
    """ Class representing Place Recognition algorithm. """

    def __init__(self, model_path, device, gallery_path, cpu_extension, gallery_size):
        self.impaths = (list(gallery_path.rglob("*.jpg")))[:gallery_size or None]
        self.model = IEModel(model_path, device, cpu_extension)
        self.input_size = self.model.input_size[2:]
        self.embeddings = self.compute_gallery_embeddings()

    def compute_embedding(self, image):
        ''' Takes input image and computes embedding vector. '''

        image = crop_resize(image, self.input_size)
        embedding = self.model.predict(image)
        return embedding

    def search_in_gallery(self, embedding):
        ''' Takes input embedding vector and searches it in the gallery. '''

        distances = np.linalg.norm(embedding - self.embeddings, axis=1, ord=2)
        sorted_indexes = np.argsort(distances)
        return sorted_indexes, distances

    def compute_gallery_embeddings(self):
        ''' Computes embedding vectors for the gallery images. '''

        images = []

        for full_path in tqdm(self.impaths, desc='Reading gallery images.'):
            image = cv2.imread(str(full_path))
            if image is None:
                print("ERROR: cannot process image, full_path =", str(full_path))
                continue
            image = crop_resize(image, self.input_size)
            images.append(image)

        embeddings = np.vstack([self.model.predict(image) for image in tqdm(
            images, desc='Computing embeddings of gallery images.')])

        return embeddings
