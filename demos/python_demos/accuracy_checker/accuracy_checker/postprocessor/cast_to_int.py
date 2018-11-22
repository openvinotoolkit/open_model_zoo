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
import numpy as np
from ..config import StringField
from ..representation import DetectionAnnotation, DetectionPrediction
from .postprocessor import Postprocessor, BasePostprocessorConfig

round_policies_func = {'nearest': np.rint,
                       'nearest_to_zero': np.trunc,
                       'lower': np.floor,
                       'greater': np.ceil}


class CastToInt(Postprocessor):
    __provider__ = 'cast_to_int'
    annotation_types = (DetectionAnnotation, )
    prediction_types = (DetectionPrediction, )

    def validate_config(self):
        class _CastToIntConfigValidator(BasePostprocessorConfig):
            round_policy = StringField(optional=True, choices=round_policies_func.keys())

        cast_to_int_config_validator = _CastToIntConfigValidator(self.__provider__)
        cast_to_int_config_validator.validate(self.config)

    def configure(self):
        self.round_func = round_policies_func[self.config.get('round_policy', 'nearest')]

    def process_image(self, annotation, prediction):
        def cast_entry(entry):
            entry.x_mins = self.round_func(entry.x_mins)
            entry.x_maxs = self.round_func(entry.x_maxs)
            entry.y_mins = self.round_func(entry.y_mins)
            entry.y_maxs = self.round_func(entry.y_maxs)

        for annotation_ in annotation:
            cast_entry(annotation_)

        for prediction_ in prediction:
            cast_entry(prediction_)

        return annotation, prediction
