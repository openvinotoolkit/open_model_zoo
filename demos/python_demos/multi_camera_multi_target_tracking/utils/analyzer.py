"""
 Copyright (c) 2019 Intel Corporation
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
import logging as log
import numpy as np
import os
from tqdm import tqdm

from tensorboardX import SummaryWriter

from utils.visualization import plot_timeline


class Analyzer(object):
    def __init__(self, cam_id, enable,
                 show_distances=True,
                 concatenate_imgs_with_distances=True,
                 plot_timeline_freq=0,
                 save_distances='',
                 save_timeline='',
                 crop_size=(32, 64)):
        self.enable = enable
        self.id = cam_id
        self.show_distances = show_distances
        self.concatenate_distances = concatenate_imgs_with_distances
        self.plot_timeline_freq = plot_timeline_freq

        self.save_distances = os.path.join(save_distances, 'sct_{}'.format(cam_id)) \
            if len(save_distances) else ''
        self.save_timeline = os.path.join(save_timeline, 'sct_{}'.format(cam_id)) \
            if len(save_timeline) else ''

        if self.save_distances and not os.path.exists(self.save_distances):
            os.makedirs(self.save_distances)
        if self.save_timeline and not os.path.exists(self.save_timeline):
            os.makedirs(self.save_timeline)

        self.dist_names = ['Latest_feature', 'Average_feature', 'Cluster_feature', 'GIoU', 'Affinity_matrix']
        self.distance_imgs = [None for _ in range(len(self.dist_names))]
        self.current_detections = []  # list of numpy arrays
        self.crop_size = crop_size  # w x h

    def prepare_distances(self, tracks, current_detections):
        tracks_num = len(tracks)
        detections_num = len(current_detections)
        w, h = self.crop_size

        target_height = detections_num + 2
        target_width = tracks_num + 2

        img_size = (
            self.crop_size[1] * target_height,
            self.crop_size[0] * target_width, 3
        )

        for j, dist_img in enumerate(self.distance_imgs):
            self.distance_imgs[j] = np.full(img_size, 225, dtype='uint8')
            dist_img = self.distance_imgs[j]
            # Insert IDs:
            # 1. Tracked objects
            for i, track in enumerate(tracks):
                id = str(track.id)
                dist_img = cv2.putText(dist_img, id, ((i + 2) * w + 5, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            # 2. Current detections
            for i, det in enumerate(current_detections):
                id = str(i)
                dist_img = cv2.putText(dist_img, id, (5, (i + 2) * h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            # Insert crops
            # 1. Tracked objects (the latest crop)
            for i, track in enumerate(tracks):
                crop = track.crops[-1]
                y0, y1, x0, x1 = h, h * 2, (i + 2) * w, (i + 2) * w + w
                dist_img[y0: y1, x0: x1, :] = crop
            # 2. Current detections
            for i, det in enumerate(current_detections):
                dist_img[(i + 2) * h: (i + 2) * h + h, w: w * 2, :] = det
            # Insert grid line
            for n, i in enumerate(range(self.crop_size[1], dist_img.shape[0] + 1, self.crop_size[1])):
                x0, y0, x1, y1 = 0, i, dist_img.shape[1] - 1, i
                x0 = self.crop_size[0] * 2 if n < 1 else x0
                cv2.line(dist_img, (x0, y0 - 1), (x1, y1 - 1), (0, 0, 0), 1, 1)
            for n, i in enumerate(range(0, dist_img.shape[1] + 1, self.crop_size[0])):
                x0, y0, x1, y1 = i, 0, i, dist_img.shape[0] - 1
                y0 = self.crop_size[1] * 2 if n == 1 else y0
                cv2.line(dist_img, (x0 - 1, y0), (x1 - 1, y1), (0, 0, 0), 1, 1)
            # Insert hat
            x0, y0, x1, y1 = 0, 0, self.crop_size[0] * 2, self.crop_size[1] * 2
            cv2.line(dist_img, (x0, y0), (x1, y1), (0, 0, 0), 1, 1)
            dist_img = cv2.putText(dist_img, 'Tracks', (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            dist_img = cv2.putText(dist_img, 'Detect', (4, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    def visualize_distances(self, id_track=0, id_det=0, distances=None, affinity_matrix=None, active_tracks_idx=None):
        w, h = self.crop_size
        if affinity_matrix is None:
            for k, dist in enumerate(distances):
                value = str(dist)[:4] if dist else ' -'
                dist_img = self.distance_imgs[k]
                position = ((id_track + 2) * w + 1, (id_det + 2) * h + 24)
                dist_img = cv2.putText(dist_img, value, position, cv2.FONT_HERSHEY_SIMPLEX, 0.41, (0, 0, 0), 1)
        else:
            dist_img = self.distance_imgs[-1]
            for i in range(affinity_matrix.shape[0]):
                for j in range(affinity_matrix.shape[1]):
                    value = str(affinity_matrix[i][j])[:4] if affinity_matrix[i][j] else ' -'
                    track_id = active_tracks_idx[j]
                    position = ((track_id + 2) * w + 1, (i + 2) * h + 24)
                    dist_img = cv2.putText(dist_img, value, position, cv2.FONT_HERSHEY_SIMPLEX, 0.41, (0, 0, 0), 1)

    def show_all_dist_imgs(self, time, active_tracks):
        if self.distance_imgs[0] is None or not active_tracks:
            return
        concatenated_dist_img = None
        if self.concatenate_distances:
            for i, img in enumerate(self.distance_imgs):
                width = img.shape[1]
                height = 32
                title = np.full((height, width, 3), 225, dtype='uint8')
                title = cv2.putText(title, self.dist_names[i], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.line(title, (0, height - 1), (width - 1, height - 1), (0, 0, 0), 1, 1)
                cv2.line(title, (width - 1, 0), (width - 1, height - 1), (0, 0, 0), 1, 1)
                img = np.vstack([title, img])
                self.distance_imgs[i] = img
            concatenated_dist_img = np.hstack([self.distance_imgs[i] for i in range(0, 3)])
            concatenated_iou_am_img = np.hstack([self.distance_imgs[i] for i in range(3, 5)])
            empty_img = np.full(self.distance_imgs[2].shape, 225, dtype='uint8')
            concatenated_iou_am_img = np.hstack([concatenated_iou_am_img, empty_img])
            concatenated_dist_img = np.vstack([concatenated_dist_img, concatenated_iou_am_img])

        if self.show_distances:
            if concatenated_dist_img is not None:
                cv2.imshow('SCT_{}_Distances'.format(self.id), concatenated_dist_img)
            else:
                for i, img in enumerate(self.distance_imgs):
                    cv2.imshow(self.dist_names[i], img)
        if len(self.save_distances):
            if concatenated_dist_img is not None:
                file_path = os.path.join(self.save_distances, 'frame_{}_dist.jpg'.format(time))
                cv2.imwrite(file_path, concatenated_dist_img)
            else:
                for i, img in enumerate(self.distance_imgs):
                    file_path = os.path.join(self.save_distances, 'frame_{}_{}.jpg'.format(time, self.dist_names[i]))
                    cv2.imwrite(file_path, img)

    def plot_timeline(self, id, time, tracks):
        if self.plot_timeline_freq > 0 and time % self.plot_timeline_freq == 0:
            plot_timeline(id, time, tracks, self.save_timeline,
                          name='SCT', show_online=self.plot_timeline_freq)


def save_embeddings(scts, save_path, use_images=False, step=0):
    def make_label_img(label_img, crop, target_size=(32, 32)):
        img = cv2.resize(crop, target_size)  # Resize, size must be square
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB
        img = np.transpose(img, (2, 0, 1)) / 255  # Scale
        label_img = np.expand_dims(img, 0) if label_img is None else \
            np.concatenate((label_img, np.expand_dims(img, 0)))
        return label_img

    summary_writer = SummaryWriter(save_path)
    embeddings_all = None
    embeddings_avg = None
    embeddings_clust = None
    metadata_avg = []
    metadata_all = []
    metadata_clust = []
    label_img_all = None
    label_img_avg = None
    label_img_clust = None
    for i, sct in enumerate(scts):
        for track in tqdm(sct.tracks, 'Processing embeddings: SCT#{}...'.format(i)):
            if use_images and len(track.crops) == 1 and track.crops[0] is None:
                log.warning('For embeddings was enabled parameter \'use_images\' but images were not found!'
                            '\'use_images\' switched off. Please check if parameter \'enable\' for analyzer'
                            'is set to True')
                use_images = False
            # Collect average embeddings
            if isinstance(track.f_avg.avg, int):
                continue
            embeddings_avg = track.f_avg.avg.reshape((1, -1)) if embeddings_avg is None else \
                np.concatenate((embeddings_avg, track.f_avg.avg.reshape((1, -1))))
            metadata_avg.append('sct_{}_'.format(i) + str(track.id))
            if use_images:
                label_img_avg = make_label_img(label_img_avg, track.crops[0])
            # Collect all embeddings
            features = None
            offset = 0
            for j, f in enumerate(track.features):
                if f is None:
                    offset += 1
                    continue
                features = f.reshape((1, -1)) if features is None else \
                    np.concatenate((features, f.reshape((1, -1))))
                metadata_all.append(track.id)
                if use_images:
                    crop = track.crops[j - offset]
                    label_img_all = make_label_img(label_img_all, crop)
            embeddings_all = features if embeddings_all is None else \
                np.concatenate((embeddings_all, features))
            # Collect clustered embeddings
            for j, f_clust in enumerate(track.f_clust.clusters):
                embeddings_clust = f_clust.reshape((1, -1)) if embeddings_clust is None else \
                                    np.concatenate((embeddings_clust, f_clust.reshape((1, -1))))
                metadata_clust.append(str(track.id))
                if use_images:
                    label_img_clust = make_label_img(label_img_clust, track.crops[j])

    summary_writer.add_embedding(embeddings_all, metadata=metadata_all,
                                 label_img=label_img_all, global_step=step, tag='All')
    summary_writer.add_embedding(embeddings_avg, metadata=metadata_avg,
                                 label_img=label_img_avg, global_step=step, tag='Average')
    summary_writer.add_embedding(embeddings_clust, metadata=metadata_clust,
                                 label_img=label_img_clust, global_step=step, tag='Clustered')
    log.info('Embeddings have been saved successfully. To see the result use the following command: '
             'tensorboard --logdir={}'.format(save_path))
    summary_writer.close()
