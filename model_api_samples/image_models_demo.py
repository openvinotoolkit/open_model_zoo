#!/usr/bin/env python3
"""
 Copyright (C) 2018-2022 Intel Corporation

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

import sys
from pathlib import Path

import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/model_tools/src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'demos/common/python'))

from openvino.model_zoo.model_api.models import Classification
from openvino.model_zoo.model_api.models import DetectionModel
from openvino.model_zoo.model_api.models import SegmentationModel

def main():
    if len(sys.argv) != 2:
        raise RuntimeError(f'Usage: {sys.argv[0]} <path_to_image>')
    
    image = cv2.imread(sys.argv[1])
    if image is None:
        raise RuntimeError('Failed to read the image')
    
    # Create Image Classification model using mode name and download from Open Model Zoo
    mobilenetv2 = Classification.create_model("efficientnet-b0-pytorch")
    classifications = mobilenetv2(image)
    print(f"Classification results: {classifications}")
    
    # Create Object Detection model using mode name and download from Open Model Zoo
    # Replace numpy preprocessing and embed it directly into a model graph to speed up inference
    ssd = DetectionModel.create_model("ssd_mobilenet_v1_fpn_coco", configuration={"embed_preprocessing": True})
    detections = ssd(image)
    print(f"Detection results: {detections}")
    
    # Create Image Segmentation model
    ssd_local = DetectionModel.create_model("/home/alex/.cache/omz/public/ssd_mobilenet_v1_fpn_coco/FP16/ssd_mobilenet_v1_fpn_coco.xml")
    detections = ssd_local(image)
    print(f"Detection results local: {detections}")
    
    # Create Image Segmentation model
    hrnet = SegmentationModel.create_model("hrnet-v2-c1-segmentation")
    mask = hrnet(image)
    Image.fromarray(mask + 20).save("mask.png")

if __name__ == '__main__':
    main()
