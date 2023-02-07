import tempfile
import requests
import os
from copy import deepcopy
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import cv2

# Temporary WA
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'tools/model_tools/src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'demos/common/python'))

from openvino.model_zoo.model_api.models import Model, Classification
from openvino.model_zoo.model_api.models import Detection


IMAFE_FILE = tempfile.NamedTemporaryFile(suffix=".onnx").name


def download_image(save_path):
    URL="https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/test_images/image1.jpg"
    if not os.path.exists(save_path):
        r = requests.get(URL, allow_redirects=True)
        open(save_path, 'wb').write(r.content)
        

def compare_output_objects(ref, obj):
    if type(ref) != type(obj):
        raise RuntimeError(f"Type of reference and object are different."
                           f"Reference: {ref}, object: {obj}")
    if isinstance(ref, tuple):
        if len(ref) != len(obj):
            raise RuntimeError(f"Length of reference and object are different."
                               f"Reference: {ref}, object: {obj}")
        # Classes should be the same but scores can be different by a small margin
        return ref[0] == obj[0] and math.isclose(ref[2], ref[2], rel_tol=1e-03)
    if isinstance(ref, Detection):
        result = True
        for ref_coord, obj_coord in zip(ref.get_coords(), obj.get_coords()):
            result = result and math.isclose(ref_coord, obj_coord, rel_tol=1.0)
        result = result and math.isclose(ref.score, obj.score, rel_tol=1e-03)
        result = result and ref.id == obj.id
        return result
    return False

        
def compare_model_outputs(references, objects):
    references_to_compare = references
    objects_to_compare = objects
    if not isinstance(references, list):
        references_to_compare = [references]
        objects_to_compare = [objects]
        
    if len(references_to_compare) != len(objects_to_compare):
        raise RuntimeError(f"Length of reference and object are different."
                            f"Reference: {references_to_compare}, object: {objects_to_compare}")
        
    result = True
    for ref, obj in zip(references_to_compare, objects_to_compare):
        result = compare_output_objects(ref, obj)
        if result is False:
            assert f"Results with embedded preprocessing does not correspond to the results without preprocessing." \
                f"Reference: {references_to_compare}, object: {objects_to_compare}"
            break
    return result
        

@pytest.mark.parametrize(("model_name"), 
            ["resnet-18-pytorch", "yolo-v4-tf"])
def test_image_models(model_name):
    download_image(IMAFE_FILE)
    
    model = Model.create_model(model_name)
    image = cv2.imread(IMAFE_FILE)
    if image is None:
        raise RuntimeError("Failed to read the image")
    ref_output = model(deepcopy(image))
    
    model_w_preprocess = Model.create_model(model_name, configuration={"embed_preprocessing": True})
    to_compare = model_w_preprocess(deepcopy(image))
    
    assert compare_model_outputs(ref_output, to_compare)

@pytest.fixture(scope="session", autouse=True)
def clean_up():
    if os.path.exists(IMAFE_FILE):
        os.remove(IMAFE_FILE)