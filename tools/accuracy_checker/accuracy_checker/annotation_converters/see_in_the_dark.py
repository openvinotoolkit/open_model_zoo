"""
Copyright (c) 2018-2021 Intel Corporation

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
from pathlib import Path
from .format_converter import FileBasedAnnotationConverter, ConverterReturn
from ..representation import ImageProcessingAnnotation
from ..representation.image_processing import GTLoader
from ..utils import read_txt, check_file_existence


class SeeInTheDarkDatasetConverter(FileBasedAnnotationConverter):
    __provider__ = 'see_in_the_dark'

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        images_list = read_txt(self.annotation_file)
        annotations = []
        content_errors = None if not check_content else []
        num_images = len(images_list)
        for idx, line in enumerate(images_list):
            input_image, gt_image = line.split(' ')[:2]
            identifier = Path(input_image).name
            gt_identifier = Path(gt_image).name
            in_exposure = float(identifier[9:-5])
            gt_exposure = float(identifier[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)
            if check_content:
                if not check_file_existence(self.annotation_file.parent / input_image):
                    content_errors.append('{}: does not exist'.format(self.annotation_file.parent / input_image))
                if not check_file_existence(self.annotation_file.parent / gt_image):
                    content_errors.append('{}: does not exist'.format(self.annotation_file.parent / gt_image))
            annotation = ImageProcessingAnnotation(identifier, gt_identifier, gt_loader=GTLoader.RAWPY)
            annotation.metadata['ratio'] = ratio
            annotations.append(annotation)
            if progress_callback and idx % progress_interval:
                progress_callback(idx * 100 / num_images)

        return ConverterReturn(annotations, None, content_errors)
