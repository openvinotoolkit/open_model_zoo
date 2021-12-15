# -*- coding: utf-8 -*-
import time
from collections import deque
from importlib.machinery import SourceFileLoader
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F

import feature_embedding
import mstcn


class Segmentor(object):
    def __init__(self, ):
        pass

    def initialize(self, opt):
        """
            (1) Initialize the embed model/graph/session and the seg model.
            (2) Load the predefined parameters.
        """
        self.opt = opt
        # Initialize the embedding buffer (ndarray)
        self.embedding_buffer1 = np.zeros((opt.embed_dim, 0))
        self.embedding_buffer2 = np.zeros((opt.embed_dim, 0))

        # ************************** Initialize the embedding model/session/graph (I3D,Tensorflow) **************************
        self.embed_model, self.embed_sess = feature_embedding.embed_model_initialize(opt=opt)
        self.embed_input = tf.placeholder(tf.float32,
                                          shape=(opt.embed_batch_size, opt.embed_window_length, opt.img_size_h, opt.img_size_w, 3))
        with tf.variable_scope('RGB'):
            rgb_model = self.embed_model
            rgb_logits, _ = rgb_model(self.embed_input, is_training=False, dropout_keep_prob=1.0)
        self.embed_logits = rgb_logits
        self.embed_predictions = tf.nn.softmax(self.embed_logits)

        # Warm_up Operation
        feed_dict = {}
        input_data = [[np.zeros((opt.img_size_h, opt.img_size_w, 3)) for j in range(opt.embed_window_length)]
                      for i in range(opt.embed_batch_size)]  # B x N x H x W x 3
        feed_dict[self.embed_input] = input_data
        out_logits, _ = self.embed_sess.run([self.embed_logits, self.embed_predictions], feed_dict=feed_dict)

        # ************************** Initialize the segmentation model (MSTCN++, Pytorch) **************************
        seg_model_path = Path(opt.seg_model_dir, "epoch-400.model")
        self.seg_model = mstcn.seg_model_initialize(model_path=str(seg_model_path), opt=opt, gpu_id=1)

        # Seg buffer initialization
        his_buffer_len = 2 ** 11
        input_0 = torch.zeros((2 * opt.embed_dim, his_buffer_len), dtype=torch.float)
        input_0.unsqueeze_(0)
        input_0 = input_0.cuda()  # 1x2048x2048
        _, self.his_fea = self.seg_model(input_0)  # [12*[1x64x2048], 11*[1x64x2048], 11*[1x64x2048], 11*[1x64x2048]]

        self.temporal_logits = np.zeros((0, opt.num_classes))  # embedding-based predictions

        if self.opt.sliding_smoothing:
            max_length = 2 ** 16
            self.temporal_predictions = np.zeros((max_length, opt.num_classes))  # frame-based predictions
            self.temporal_counter = np.zeros((max_length))  # frame-based inference counter

    def inference(self, buffer_top, buffer_front, frame_index):
        """
            (1) Given the buffers of the input image arrays, generate the chunk for feature embedding
                 and save them into the predefined buffer (two view simultaneously)
            (2) Given the buffers of the feature embeddings, conduct clip-based action recognition.
            (3) Dynamically record and update the final prediction results. ( provided for scale evaluation module)
        Args:
            buffer_top: buffers of the input image arrays for the top view
            buffer_front: buffers of the input image arrays for the front view
            frame_index: frame index of the latest frame (1,2,3, ...)

        Returns: the temporal prediction results for each frame (including the historical predictions)，
                 length of predictions == frame_index()

        """
        time1 = time.time()
        self.embedding_buffer1 = self.feature_embedding(img_buffer=buffer_top,
                                                        embedding_buffer=self.embedding_buffer1,
                                                        frame_index=frame_index, opt=self.opt)
        self.embedding_buffer2 = self.feature_embedding(img_buffer=buffer_front,
                                                        embedding_buffer=self.embedding_buffer2,
                                                        frame_index=frame_index, opt=self.opt)
        time2 = time.time()
        print("Embed time:", time2 - time1)
        if min(self.embedding_buffer1.shape[-1], self.embedding_buffer2.shape[-1]) > 0:
            self.action_segmentation(opt=opt)
            time3 = time.time()
            print("Seg time:", time3 - time2)

        frame_predictions = self.generate_frame_predictions()
        return frame_predictions

    def feature_embedding(self, img_buffer, embedding_buffer, frame_index, opt):

        window = opt.embed_window_length
        stride = opt.embed_window_stride
        atrous_rate = opt.embed_window_atrous_rate
        batch_size = opt.embed_batch_size  # default to be 1

        print("Frame embedding:", frame_index)
        min_t = 0 + stride * 0 + (window - 1) * atrous_rate  # minimal temporal length for processor

        if frame_index > min_t:
            num_embedding = embedding_buffer.shape[-1]
            img_buffer = list(img_buffer)
            feed_dict = {}
            while (0 + stride * num_embedding + (window - 1) * atrous_rate) < frame_index:
                start_index = 0 + stride * num_embedding  # absolute index in temporal shaft
                if frame_index > len(img_buffer):
                    start_index = start_index - (frame_index - len(img_buffer))  # absolute index in buffer shaft
                input_data = [[cv2.resize(img_buffer[start_index + i * atrous_rate], (opt.img_size_h, opt.img_size_w))
                               for i in range(window)] for j in range(batch_size)]

                feed_dict[self.embed_input] = input_data
                out_logits, _ = self.embed_sess.run([self.embed_logits, self.embed_predictions], feed_dict=feed_dict)
                embedding = np.array(out_logits).T  # 1024 x 1

                embedding_buffer = np.concatenate([embedding_buffer, embedding], axis=1)  # ndarray: C x num_embedding
                num_embedding += 1

        return embedding_buffer

    def action_segmentation(self, opt):

        embedding_buffer1 = self.embedding_buffer1
        embedding_buffer2 = self.embedding_buffer2
        start_index = self.temporal_logits.shape[0]
        end_index = min(embedding_buffer1.shape[-1], embedding_buffer2.shape[-1])
        batch_size = opt.seg_batch_size

        if end_index > start_index:
            print("Temporal classification ...")
            num_batch = (end_index - start_index) // batch_size
            if num_batch == 0:
                pass
            else:
                pass
                for batch_idx in range(num_batch):
                    embedding_unit1 = embedding_buffer1[:,
                                      start_index + batch_idx * batch_size:start_index + batch_idx * batch_size + batch_size]
                    embedding_unit2 = embedding_buffer2[:,
                                      start_index + batch_idx * batch_size:start_index + batch_idx * batch_size + batch_size]
                    feature_unit = np.concatenate([embedding_unit1[:, ], embedding_unit2[:, ]], axis=0)

                    input = torch.tensor(feature_unit, dtype=torch.float).cuda()  # 2048x24
                    input = input.unsqueeze_(0)

                    predictions, self.his_fea = self.seg_model(input, self.his_fea)

                    """
                        predictions --> 4x1x64x24
                        his_fea --> [12*[1x64x2048], 11*[1x64x2048], 11*[1x64x2048], 11*[1x64x2048]]
                    """
                    temporal_logits = predictions[:, :, :self.opt.num_classes, :]  # 4x1x16x24
                    temporal_logits = F.softmax(temporal_logits[-1], 1)  # 1x16x24
                    temporal_logits = temporal_logits.permute(0, 2, 1).squeeze()
                    temporal_logits = temporal_logits.detach().cpu().numpy()  # 24x16
                    # temporal_probs, temporal_predictions = torch.max(temporal_logits.data, 1)
                    self.temporal_logits = np.concatenate([self.temporal_logits, temporal_logits], axis=0)
                    if self.opt.sliding_smoothing:
                        for i in range(batch_size):
                            embed_idx = start_index + batch_idx * batch_size + i
                            time_start_idx = 0 + embed_idx * self.opt.embed_window_stride
                            time_end_idx = time_start_idx + (self.opt.embed_window_length - 1) * self.opt.embed_window_atrous_rate
                            self.temporal_counter[time_start_idx:time_end_idx + 1] += 1  # frame-based inference counter
                            self.temporal_predictions[time_start_idx:time_end_idx + 1] = temporal_logits[i]  # frame-based predictions

        else:
            print("Waiting for the next frame ...")

    def mstcn_inference(self):
        pass

    def generate_frame_predictions(self):
        if not self.opt.sliding_smoothing:
            valid_index = self.temporal_logits.shape[0]
            if valid_index == 0:
                return []
            else:
                frame_predictions = [self.opt.mapping_dict[str(i)] for i in np.argmax(self.temporal_logits, axis=1)]
                frame_predictions = ["background" for i in range(self.opt.embed_window_length - 1)] + frame_predictions
                return frame_predictions
        else:
            valid_index = np.argwhere(self.temporal_counter == 0)[0].item()
            if valid_index == 0:
                return []
            else:
                frame_predictions = self.temporal_predictions[:valid_index] / np.expand_dims(self.temporal_counter[:valid_index], 1)
                frame_predictions = [self.opt.mapping_dict[str(i)] for i in np.argmax(frame_predictions, axis=1)]
                return frame_predictions


if __name__ == "__main__":
    print(__file__)
    current_dir = Path(__file__).parent
    config_file = Path(current_dir, "config.py")
    opt = SourceFileLoader("", str(config_file)).load_module().config

    segmentor = Segmentor()
    segmentor.initialize(opt=opt)

    frame_counter = 0  # Frame index counter
    buffer1 = deque(maxlen=1000)  # Array buffer
    buffer2 = deque(maxlen=1000)
    cap1 = cv2.VideoCapture("/disk1/dataset/ping/mythwaredata/raw_video/P03_A5130001992103255012_2021-10-18_10-19-30_1.mp4")
    cap2 = cv2.VideoCapture("/disk1/dataset/ping/mythwaredata/raw_video/P03_A5130001992103255012_2021-10-18_10-19-30_2.mp4")

    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()  # frame:480 x 640 x 3
        ret2, frame2 = cap2.read()  # frame:480 x 640 x 3
        if ret1 and ret2:
            buffer1.append(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
            buffer2.append(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
            frame_counter += 1

            frame_predictions = segmentor.inference(buffer_top=buffer1, buffer_front=buffer2, frame_index=frame_counter)
            # print("Frame predictions:", frame_predictions)

        else:
            print("Finished!")
            frame_predictions = segmentor.generate_frame_predictions()
            print("Frame predictions:", frame_predictions)

            break
