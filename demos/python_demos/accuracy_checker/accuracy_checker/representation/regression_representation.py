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
from .base_representation import BaseRepresentation

class RegressionRepresentation(BaseRepresentation):
    def __init__(self, identifier='', value=None):
        super().__init__(identifier)
        self.value = value


class RegressionAnnotation(RegressionRepresentation):
    pass


class RegressionPrediction(RegressionRepresentation):
    pass


class PointRegressionRepresentation(BaseRepresentation):
    def __init__(self, identifier='', x_values=None, y_values=None):
        super().__init__(identifier)
        self.x_values = x_values if x_values.any() else []
        self.y_values = y_values if y_values.any() else []


class PointRegressionAnnotation(PointRegressionRepresentation):
    pass


class PointRegressionPrediction(PointRegressionRepresentation):
    pass
