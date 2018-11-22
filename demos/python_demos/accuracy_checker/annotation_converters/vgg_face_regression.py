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
import csv
import numpy as np
from accuracy_checker.representation.regression_representation import PointRegressionAnnotation
from accuracy_checker.utils import check_exists
from .format_converter import BaseFormatConverter


class LandmarksRegression(BaseFormatConverter):
    __provider__ = "landmarks_regression"

    def convert(self, landmarks_csv, bbox_csv=None):
        annotations = []
        check_exists(landmarks_csv)
        with open(landmarks_csv, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                identifier = row['NAME_ID']+'.jpg'
                x_values = np.array([float(row["P1X"]), float(row["P2X"]), float(row["P3X"]),
                                     float(row["P4X"]), float(row["P5X"])])
                y_values = np.array([float(row["P1Y"]), float(row["P2Y"]), float(row["P3Y"]),
                                     float(row["P4Y"]), float(row["P5Y"])])
                annotations.append(PointRegressionAnnotation(identifier, x_values, y_values))
        if bbox_csv is not None:
            check_exists(bbox_csv)
            with open(bbox_csv) as file:
                reader = csv.DictReader(file)
                for index, row in enumerate(reader):
                    x_min = int(row["X"])
                    y_min = int(row["Y"])
                    x_max = x_min + int(row["W"])
                    y_max = y_min + int(row["H"])
                    annotations[index].metadata['rect'] = [x_min, y_min, x_max, y_max]
        return annotations, {'label_map': {0: 'Left Eye', 1: 'Right Eye', 2: 'Nose',
                                           3: 'Left Mouth Corner', 4: 'Right Mouth Corner'}}
