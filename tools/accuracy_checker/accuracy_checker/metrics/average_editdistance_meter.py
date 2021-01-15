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


class AverageEditdistanceMeter:
    def __init__(self, loss):
        self.loss = loss
        self.total = 0
        self.nchars = 0
        self.curr_err_rate = 0

    def update(self, annotation_val, prediction_val):
        loss = self.loss(prediction_val, annotation_val)
        self.total += loss
        self.nchars += len(annotation_val)
        self.curr_err_rate = self.total / self.nchars

        return 1 - self.curr_err_rate

    def evaluate(self):
        self.final_err_rate = self.total / self.nchars

        return 1 - self.final_err_rate

    def reset(self):
        self.total = 0
        self.nchars = 0
