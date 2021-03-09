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

from ..representation import HitRatioAnnotation
from ..utils import read_txt, get_path
from ..config import PathField, NumberField

from .format_converter import BaseFormatConverter, ConverterReturn


class MovieLensConverter(BaseFormatConverter):
    __provider__ = "movie_lens_converter"
    annotation_types = (HitRatioAnnotation, )

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update({
            'rating_file': PathField(description="Path to rating file."),
            'negative_file': PathField(description="Path to negative file."),
            'users_max_number': NumberField(
                optional=True, min_value=1, value_type=int, description="Max number of users."
            )
        })

        return configuration_parameters

    def configure(self):
        self.rating_file = self.get_value_from_config('rating_file')
        self.negative_file = self.get_value_from_config('negative_file')
        self.users_max_number = self.get_value_from_config('users_max_number')

    def convert(self, check_content=False, **kwargs):
        annotations = []
        users = []

        for file_row in read_txt(self.rating_file):
            user_id, item_id, _ = file_row.split()
            users.append(user_id)
            identifier = ['u:'+user_id, 'i:' + item_id]
            annotations.append(HitRatioAnnotation(identifier))
            if self.users_max_number and len(users) == self.users_max_number:
                break

        item_numbers = 1

        items_neg = []
        with get_path(self.negative_file).open() as content:
            for file_row in content:
                items = file_row.split()
                items_neg.append(items)
                if self.users_max_number and len(items_neg) == self.users_max_number:
                    break

        if items_neg:
            iterations = len(items_neg[0])
            item_numbers += iterations
            for i in range(iterations):
                for user in users:
                    item = items_neg[int(user)][i]
                    identifier = ['u:' + user, 'i:' + item]
                    annotations.append(HitRatioAnnotation(identifier, False))

        return ConverterReturn(annotations, {'users_number': len(users), 'item_numbers': item_numbers}, None)
