import cv2
import logging
import threading
from collections import deque

from .async_pipeline import AsyncPipeline


class MtcnnPipeline:
    def __init__(self, ie, proposal_model, refine_model, output_model, plugin_config, device='CPU', max_num_requests=1):
        self.proposal_model = proposal_model
        self.refine_model = refine_model
        self.output_model = output_model
        self.logger = logging.getLogger()

        self.device = device
        self.max_requests = max_num_requests
        self.plugin_config = plugin_config
        self.ie = ie

        self.pipeline = AsyncPipeline(ie, self.proposal_model, plugin_config, device, max_num_requests)

    def submit_data(self, inputs, id, meta):
        pass

    def infer(self, frame):
        scales = self.proposal_model.calc_scales(frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print(*scales)
        proposal_results = []
        for id, scale in enumerate(scales):
            hs, ws = frame.shape[:2]
            hs = int(scale * hs)
            ws = int(scale * ws)
            self.proposal_model.reshape([1, 3, ws, hs])
            self.proposal_model.scale = scale
            pipeline = AsyncPipeline(self.ie, self.proposal_model, self.plugin_config, self.device, self.max_requests)
            pipeline.submit_data(rgb_frame, id, {'frame': frame})
            pipeline.await_any()
            results = pipeline.get_result(id)
            if results:
                proposal_results += results[0][0]
        return self.proposal_model.postprocess_all(proposal_results)
