"""
Copyright (c) 2018-2020 Intel Corporation

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

from .dependency import ClassProvider

class Topology(ClassProvider):
    __provider_type__ = 'topology_type'

    @classmethod
    def parameters(cls):
        return {}

class GenericTopology(Topology):
    __provider__ = 'generic_topology'

class ImageClassification(GenericTopology):
    __provider__ = 'image_classification'

class ObjectDetection(GenericTopology):
    __provider__ = 'object_detection'

class SSD(ObjectDetection):
    __provider__ = 'ssd'

class FasterRCNN(ObjectDetection):
    __provider__ = 'faster_rcnn'

class Yolo(ObjectDetection):
    __provider__ = 'yolo'

class YoloV1Tiny(Yolo):
    __provider__ = 'yolo_v1_tiny'

class YoloV2(Yolo):
    __provider__ = 'yolo_v2'

class YoloV2Tiny(Yolo):
    __provider__ = 'yolo_v2_tiny'

class YoloV3(Yolo):
    __provider__ = 'yolo_v3'

class YoloV3Tiny(Yolo):
    __provider__ = 'yolo_v3_tiny'
