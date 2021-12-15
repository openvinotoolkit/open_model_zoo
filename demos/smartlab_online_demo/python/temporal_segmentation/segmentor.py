# -*- coding: utf-8 -*-
import cv2
import torch
import numpy as np
from multiview_mobilenetv3_tsm import create_mbv3s_model

class Segmentor(object):
    def __init__(self, ):
        self.embed_model = 0
        self.seg_model = 0
        self.temporal_predictions = 0

        self.backbone_path = "<path>/<to>/<weight>"
        self.classifier_path = "<path>/<to>/<weight>"

        self.terms = [
            "noise_action",
            "put_take",
            "adjust_rider"
            ]


    def initialize(self):
        """
            TODO Initialize the embed model and the seg model.
            TODO Load the predefined parameters.
        """

        self.device = torch.device('cpu')
        # ### Torch model
        self.backbone, self.classifier = create_mbv3s_model(
            img_channel=3, n_class=3, n_view=2, fusion_op='concat')
        self.backbone, self.classifier = self.backbone.to(self.device), self.classifier.to(self.device)
        # load weight: TODO configration parse
        self.backbone.load_state_dict(.load(
            self.backbone_path,map_location=torch.device(self.device)))
        self.classifier.load_state_dict(torch.load(
            self.classifier_path,map_location=torch.device(self.device)))

    def inference(self, buffer_top, buffer_front, frame_index):
        """
            TODO Given the buffers of the input image arrays, generate the chunk for feature embedding
                 and save them into the predefined buffer (two view simultaneously)
            TODO Given the buffers of the feature embeddings, conduct clip-based action recognition.
            TODO Dynamically record and update the final prediction results. ( provided for scale evaluation module)
        Args:
            buffer_top: buffers of the input image arrays for the top view
            buffer_front: buffers of the input image arrays for the front view
            frame_index: frame index of the latest frame

        Returns: the temporal prediction results for each frame (including the historical predictions)，
                 length of predictions == frame_index()

        """

        ### preprocess ###
        buffer_front = cv2.resize(buffer_front,(224,224), interpolation= cv2.INTER_LINEAR)
        buffer_top = cv2.resize(buffer_top,(224,224), interpolation= cv2.INTER_LINEAR)
        buffer_front = buffer_front/255
        buffer_top = buffer_top/255
        
        buffer_front = torch.from_numpy(
            buffer_front[torch.newaxis,:,:,:].transpose((0,3,1,2)).astype(torch.float32)).to(self.device)
        buffer_top = torch.from_numpy(
            buffer_top[torch.newaxis,:,:,:].transpose((0,3,1,2)).astype(torch.float32)).to(self.device)

        ### run ###
        feature_vector_high = self.backbone(buffer_front)
        feature_vector_top = self.backbone(buffer_top)
        features = [feature_vector_high, feature_vector_top]
        output_torch = self.classifier(*features)

        ### yoclo classifier ###
        isAction = (output_torch.detach().cpu().numpy()[:,0] >= .5).astype(int)
        predicted = isAction * (np.argmax(output_torch.detach().cpu().numpy()[:,1:], axis=1) + 1)

        # action label will be returned e.g "adjust_rider"
        return self.terms[predicted]
