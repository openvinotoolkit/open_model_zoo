"""
Copyright (c) 2018-2022 Intel Corporation

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

import abc
import pickle # nosec - disable B403:import-pickle check


class BaseRepresentation(abc.ABC):
    def __init__(self, identifier, metadata=None):
        self.identifier = identifier
        self.metadata = metadata or {}

    @classmethod
    def load(cls, file, loader=None):
        if loader is None:
            loader = pickle.Unpickler(file) # nosec - disable B301:pickle check
        obj = loader.load()

        if cls != BaseRepresentation:
            assert isinstance(obj, cls)

        return obj

    def dump(self, file):
        pickle.dump(self, file)

    def set_image_size(self, image_sizes):
        self.metadata['image_size'] = image_sizes

    def set_data_source(self, data_source):
        self.metadata['data_source'] = data_source

    def set_segmentation_mask_source(self, mask_source):
        self.metadata['segmentation_masks_source'] = mask_source

    def set_additional_data_source(self, source):
        self.metadata['additional_data_source'] = source

    def set_dataset_metadata(self, source):
        self.metadata['dataset_meta'] = source
