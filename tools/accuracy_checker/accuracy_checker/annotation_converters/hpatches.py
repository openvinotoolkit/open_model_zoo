"""
Copyright (c) 2024 Intel Corporation

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

import os
from pathlib import Path
import numpy as np
from .format_converter import DirectoryBasedAnnotationConverter, ConverterReturn
from ..config import NumberField, StringField
from ..representation import ImageFeatureAnnotation
from ..utils import UnsupportedPackage
from ..data_readers import AnnotationDataIdentifier
from ..progress_reporters import TQDMReporter


# Large images that were ignored in previous papers
ignored_scenes = (
    "i_contruction",
    "i_crownnight",
    "i_dc",
    "i_pencils",
    "i_whitebuilding",
    "v_artisans",
    "v_astronautis",
    "v_talent",
)


class HpatchesConverter(DirectoryBasedAnnotationConverter):
    __provider__ = 'hpatches_with_kornia_feature'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'sequences_dir_name': StringField(
                optional=True, default='hpatches-sequences-release',
                description="Dataset subfolder name, where hpatches sequences are located."
            ),
            'max_num_keypoints': NumberField(
                optional=True, default=512, value_type=int, min_value=128, max_value=2048,
                description='Maksimum number of image keypoints.'
            ),
            'image_side_size': NumberField(
                optional=True, default=480, value_type=int, min_value=128, max_value=2048,
                description='Image short side size.'
            )
        })

        return params

    def configure(self):
        try:
            import torch  # pylint: disable=import-outside-toplevel
            self._torch = torch
        except ImportError as torch_import_error:
            UnsupportedPackage('torch', torch_import_error.msg).raise_error(self.__provider__)
        try:
            import kornia # pylint: disable=import-outside-toplevel
            self._kornia = kornia
        except ImportError as kornia_import_error:
            UnsupportedPackage('kornia', kornia_import_error.msg).raise_error(self.__provider__)


        self.data_dir = self.get_value_from_config('data_dir')
        self.sequences_dir = self.get_value_from_config('sequences_dir_name')
        self.max_num_keypoints = self.get_value_from_config('max_num_keypoints')
        self.side_size = self.get_value_from_config('image_side_size')

    def _get_new_image_size(self, h: int, w: int):
        side_size = self.side_size
        aspect_ratio = w / h
        if aspect_ratio < 1.0:
            size = int(side_size / aspect_ratio), side_size
        else:
            size = side_size, int(side_size * aspect_ratio)
        return size


    def _get_image_data(self, path, image_size = None):
        img = self._kornia.io.load_image(path, self._kornia.io.ImageLoadType.RGB32, device='cpu')[None, ...]

        h, w = img.shape[-2:]
        size = h, w
        size = self._get_new_image_size(h, w)
        if image_size and size != image_size:
            size = image_size
        img = self._kornia.geometry.transform.resize(
            img,
            size,
            side='short',
            antialias=True,
            align_corners=None,
            interpolation='bilinear',
        )
        scale = self._torch.Tensor([img.shape[-1] / w, img.shape[-2] / h]).to(img)
        T = np.diag([scale[0], scale[1], 1])

        data = {
            "scales": scale,
            "image_size": np.array(size[::-1]),
            "transform": T,
            "original_image_size": np.array([w, h]),
            "image" : img
        }
        return data

    @staticmethod
    def _read_homography(path):
        with open(path, encoding="utf-8") as f:
            result = []
            for line in f.readlines():
                while "  " in line:  # Remove double spaces
                    line = line.replace("  ", " ")
                line = line.replace(" \n", "").replace("\n", "")
                # Split and discard empty strings
                elements = list(filter(lambda s: s, line.split(" ")))
                if elements:
                    result.append(elements)
            return np.array(result).astype(float)

    def get_image_features(self, model, data):
        with self._torch.inference_mode():
            return model(data["image"], self.max_num_keypoints, pad_if_not_divisible=True)[0]

    def convert(self, check_content=False, progress_callback=None, progress_interval=50, **kwargs):
        annotations = []
        items = []

        sequences_dir = Path(os.path.join(self.data_dir, self.sequences_dir))
        sequences = sorted([x.name for x in sequences_dir.iterdir()])

        for seq in sequences:
            if seq in ignored_scenes:
                continue
            for i in range(2, 7):
                items.append((seq, i, seq[0] == "i"))

        disk_model = self._kornia.feature.DISK().from_pretrained("depth")

        num_iterations = len(items)
        progress_reporter = TQDMReporter(print_interval=progress_interval)
        progress_reporter.reset(num_iterations)

        for item_id, item in enumerate(items):
            seq, idx, _ = item

            if idx == 2:
                img_path = Path(sequences_dir / seq / "1.ppm")
                data0 = self._get_image_data(img_path)
                features0 = self.get_image_features(disk_model, data0)

            img_path = Path(sequences_dir / seq / f"{idx}.ppm")
            data1 = self._get_image_data(img_path)
            features1 = self.get_image_features(disk_model, data1)

            H = self._read_homography(Path(sequences_dir / seq / f"H_1_{idx}"))
            H = data1["transform"] @ H @ np.linalg.inv(data0["transform"])

            data = {
                "keypoints0": features0.keypoints.unsqueeze(0),
                "keypoints1": features1.keypoints.unsqueeze(0),
                "descriptors0": features0.descriptors.unsqueeze(0),
                "descriptors1" : features1.descriptors.unsqueeze(0),
                "image_size0": data0["image_size"],
                "image_size1": data1["image_size"],
                "H_0to1": H
            }

            sequence = f"{seq}/{idx}"
            annotated_id = AnnotationDataIdentifier(sequence, data)
            annotation = ImageFeatureAnnotation(
                identifier = annotated_id,
                sequence = sequence
            )
            annotations.append(annotation)
            progress_reporter.update(item_id, 1)

        progress_reporter.finish()
        return ConverterReturn(annotations, None, None)
