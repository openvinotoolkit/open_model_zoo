"""
Copyright (c) 2018-2024 Intel Corporation

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
from ..utils import UnsupportedPackage
from .metric import Metric
from ..config import StringField, BoolField
from ..representation import Text2ImageGenerationAnnotation, Text2ImageGenerationPrediction, ImageProcessingPrediction


class ClipScore(Metric):
    __provider__ = 'clip_score'
    annotation_types = (Text2ImageGenerationAnnotation, )
    prediction_types = (ImageProcessingPrediction, Text2ImageGenerationPrediction)

    def __init__(self, *args, **kwargs):
        try:
            from torchmetrics.functional.multimodal.clip_score import clip_score # pylint: disable=C0415
            self.clip_score = clip_score
        except ImportError as _import_err:
            self.clip_score = UnsupportedPackage('clip_score', _import_err)

        super().__init__(*args, **kwargs)
        self._score = 0
        self._num_images = 0

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'clip_model_name': StringField(
            default="openai/clip-vit-base-patch16", optional=True, description="CLIP model for evaluation"
            ),
            "normalized_image": BoolField(
            optional=True, default=False, description="Provided image in [0, 1] floating point range"
            ),
            "channel_order": StringField(
            choices=["BGR", "RGB"], default="RGB", optional=True, description="Channel order for image"
            )
        })

        return parameters

    def configure(self):
        if isinstance(self.clip_score, UnsupportedPackage):
            self.clip_score.raise_error(self.__provider__)
        self._model_name = self.get_value_from_config("clip_model_name")
        self.normalized_image = self.get_value_from_config("normalized_image")
        self.channle_order = self.get_value_from_config("channel_order")

    def reset(self):
        self._score = 0
        self._num_images = 0

    def evaluate(self, annotations, predictions):
        if self._num_images == 0:
            return 0
        return self._score / self._num_images

    def update(self, annotation, prediction):
        prompt = annotation.prompt
        image = self._to_torch(prediction.value)
        current_score = self.clip_score(image, [prompt], model_name_or_path=self._model_name)
        self._score += current_score.detach().numpy()
        self._num_images += 1
        return current_score.detach().numpy()

    def _to_torch(self, image):
        import torch # pylint: disable=C0415
        if self.normalized_image:
            image *= 255
        image = image.astype(np.uint8)
        if self.channle_order == "BGR":
            image = image[:, :, ::-1]
        image_tensor = np.transpose(image, (2, 0, 1))
        image_tensor = np.expand_dims(image_tensor, 0)
        tensor = torch.from_numpy(image_tensor)
        return tensor

    @classmethod
    def get_common_meta(cls):
        meta = super().get_common_meta()
        meta['scale'] = 1
        meta['postfix'] = ''
        return meta
