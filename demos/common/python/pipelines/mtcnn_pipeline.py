"""
 Copyright (C) 2021 Intel Corporation

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

import cv2
import logging

from .async_pipeline import AsyncPipeline
from .sync_pipeline import SyncPipeline


class MtcnnPipeline:
    def __init__(self, ie, proposal_model, refine_model, output_model,
                 pm_sync=True, pm_device='CPU', pm_config={},
                 rm_batch_size=0, rm_device='CPU', rm_config={}, rm_num_requests=1,
                 om_batch_size=0, om_device='CPU', om_config={}, om_num_requests=1):
        self.proposal_model = proposal_model
        self.refine_model = refine_model
        self.output_model = output_model
        self.logger = logging.getLogger()

        self.ie = ie

        self.proposal_sync = pm_sync
        self.proposal_device = pm_device
        self.proposal_config = pm_config

        self.refine_resize = rm_batch_size == 0
        self.refine_fixed_batch_size = rm_batch_size
        if self.refine_resize:
            self.refine_pipeline = SyncPipeline(ie, device=rm_device, silent=True)
        else:
            self.refine_model.set_batch_size(self.refine_fixed_batch_size)
            self.refine_pipeline = AsyncPipeline(ie, self.refine_model, rm_config, rm_device, rm_num_requests,
                                                 silent=True)

        self.output_resize = om_batch_size == 0
        self.output_fixed_batch_size = om_batch_size
        if self.output_resize:
            self.output_pipeline = SyncPipeline(ie, device=om_device, silent=True)
        else:
            self.output_model.set_batch_size(self.output_fixed_batch_size)
            self.output_pipeline = AsyncPipeline(ie, self.output_model, om_config, om_device, om_num_requests,
                                                 silent=True)

    def infer(self, frame):
        scales = self.proposal_model.calc_scales(frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        proposal_results = []
        if self.proposal_sync:
            for id, scale in enumerate(scales):
                hs = int(scale * h)
                ws = int(scale * w)
                self.proposal_model.reshape([1, 3, ws, hs])
                self.proposal_model.scale = scale
                pipeline = SyncPipeline(self.ie, self.proposal_model, self.proposal_device)
                proposal_results += pipeline.infer(rgb_frame)
        else:
            proposal_pipelines = []
            for id, scale in enumerate(scales):
                hs = int(scale * h)
                ws = int(scale * w)
                self.proposal_model.reshape([1, 3, ws, hs])
                self.proposal_model.scale = scale
                proposal_pipeline = AsyncPipeline(self.ie, self.proposal_model, plugin_config=self.proposal_config,
                                                  device=self.proposal_device, max_num_requests=1, silent=True)
                proposal_pipeline.submit_data(rgb_frame, 0, {'frame': frame})
                proposal_pipelines.append(proposal_pipeline)
            while len(proposal_pipelines) != 0:
                proposal_pipeline = proposal_pipelines.pop(0)
                if proposal_pipeline.has_completed_request():
                    results = proposal_pipeline.get_result(0)
                    if results:
                        proposal_results += results[0][0]
                else:
                    proposal_pipelines.append(proposal_pipeline)
        proposal_results = self.proposal_model.postprocess_all(proposal_results)

        if len(proposal_results) == 0:
            return []

        if self.refine_resize:
            self.refine_model.set_batch_size(len(proposal_results))
            self.refine_pipeline.reload_model(self.refine_model)
            refine_results = self.refine_pipeline.infer({'image': rgb_frame, 'crop': proposal_results})
        else:
            refine_results = []
            while len(proposal_results) != 0:
                results = self.refine_pipeline.get_result()
                if results:
                    refine_results += results[0][0]
                if self.refine_pipeline.is_ready():
                    detection = []
                    while len(detection) < self.refine_fixed_batch_size and len(proposal_results) > 0:
                        detection.append(proposal_results.pop())
                    self.refine_pipeline.submit_data({'image': rgb_frame, 'crop': detection}, len(proposal_results), {})
                else:
                    self.refine_pipeline.await_any()
            self.refine_pipeline.await_all()
            while self.refine_pipeline.has_completed_request():
                results = self.refine_pipeline.get_result()
                if results:
                    refine_results += results[0][0]
        refine_results = self.refine_model.postprocess_all(refine_results)

        if len(refine_results) == 0:
            return []

        if self.output_resize:
            self.output_model.set_batch_size(len(refine_results))
            self.output_pipeline.reload_model(self.output_model)
            output_results = self.output_pipeline.infer({'image': rgb_frame, 'crop': refine_results})
        else:
            output_results = []
            while len(refine_results) != 0:
                results = self.output_pipeline.get_result()
                if results:
                    output_results += results[0][0]
                if self.output_pipeline.is_ready():
                    detection = []
                    while len(detection) < self.output_fixed_batch_size and len(refine_results) > 0:
                        detection.append(refine_results.pop())
                    self.output_pipeline.submit_data({'image': rgb_frame, 'crop': detection}, len(refine_results), {})
                else:
                    self.output_pipeline.await_any()
            self.output_pipeline.await_all()
            while self.output_pipeline.has_completed_request():
                results = self.output_pipeline.get_result()
                if results:
                    output_results += results[0][0]
        output_results = self.output_model.postprocess_all(output_results)

        output_results = self.output_model.make_detections(output_results)
        return output_results
