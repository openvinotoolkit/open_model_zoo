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

import argparse
import json
import logging as log

import motmetrics as mm
import numpy as np
from xml.etree import ElementTree as etree
from tqdm import tqdm

from mc_tracker.sct import TrackedObj
from utils.misc import set_log_config

set_log_config()


def read_gt_tracks(gt_filenames, size_divisor=1, skip_frames=0, skip_heavy_occluded_objects=False):
    min_last_frame_idx = -1
    camera_tracks = [[] for _ in gt_filenames]
    for i, filename in enumerate(gt_filenames):
        last_frame_idx = -1
        tree = etree.parse(filename)
        root = tree.getroot()
        for track_xml_subtree in tqdm(root, desc='Reading ' + filename):
            if track_xml_subtree.tag != 'track':
                continue
            track = {'id': None, 'boxes': [], 'timestamps': []}
            for box_tree in track_xml_subtree.findall('box'):
                if skip_frames > 0 and int(box_tree.get('frame')) % skip_frames == 0:
                    continue
                occlusion = [tag.text for tag in box_tree if tag.attrib['name'] == 'occlusion'][0]
                if skip_heavy_occluded_objects and occlusion == 'heavy_occluded':
                    continue
                x_left = int(float(box_tree.get('xtl'))) // size_divisor
                x_right = int(float(box_tree.get('xbr'))) // size_divisor
                y_top = int(float(box_tree.get('ytl'))) // size_divisor
                y_bottom = int(float(box_tree.get('ybr'))) // size_divisor
                assert x_right > x_left
                assert y_bottom > y_top
                track['boxes'].append([x_left, y_top, x_right, y_bottom])
                track['timestamps'].append(int(box_tree.get('frame')) // size_divisor)
                last_frame_idx = max(last_frame_idx, track['timestamps'][-1])
                id = [int(tag.text) for tag in box_tree if tag.attrib['name'] == 'id'][0]
            track['id'] = id
            camera_tracks[i].append(track)
        if min_last_frame_idx < 0:
            min_last_frame_idx = last_frame_idx
        else:
            min_last_frame_idx = min(min_last_frame_idx, last_frame_idx)

    return camera_tracks, min_last_frame_idx


def get_detections_from_tracks(tracks_history, time):
    active_detections = [[] for _ in tracks_history]
    for i, camera_hist in enumerate(tracks_history):
        for track in camera_hist:
            if time in track['timestamps']:
                idx = track['timestamps'].index(time)
                active_detections[i].append(TrackedObj(track['boxes'][idx], track['id']))
    return active_detections


def check_contain_duplicates(all_detections):
    for detections in all_detections:
        all_labels = [obj.label for obj in detections]
        uniq = set(all_labels)
        if len(all_labels) != len(uniq):
            return True

    return False


def main():
    """Computes MOT metrics for the multi camera multi person tracker"""
    parser = argparse.ArgumentParser(description='Multi camera multi person \
                                                  tracking visualization demo script')
    parser.add_argument('--history_file', type=str, default='', required=True,
                        help='File with tracker history')
    parser.add_argument('--gt_files', type=str, nargs='+', required=True,
                        help='Files with ground truth annotation')
    parser.add_argument('--size_divisor', type=int, default=1,
                        help='Scale factor for GT image resolution')
    parser.add_argument('--skip_frames', type=int, default=0,
                        help='Frequency of skipping frames')

    args = parser.parse_args()

    with open(args.history_file) as hist_f:
        history = json.load(hist_f)

    assert len(args.gt_files) == len(history)
    gt_tracks, last_frame_idx = read_gt_tracks(args.gt_files,
                                               size_divisor=args.size_divisor,
                                               skip_frames=args.skip_frames)
    accs = [mm.MOTAccumulator(auto_id=True) for _ in args.gt_files]

    for time in tqdm(range(last_frame_idx + 1), 'Processing detections'):
        active_detections = get_detections_from_tracks(history, time)
        if check_contain_duplicates(active_detections):
            log.info('Warning: at least one IDs collision has occured at the timestamp ' + str(time))
        gt_detections = get_detections_from_tracks(gt_tracks, time)

        for i, camera_gt_detections in enumerate(gt_detections):
            gt_boxes = []
            gt_labels = []
            for obj in camera_gt_detections:
                gt_boxes.append([obj.rect[0], obj.rect[1],
                                 obj.rect[2] - obj.rect[0],
                                 obj.rect[3] - obj.rect[1]])
                gt_labels.append(obj.label)

            ht_boxes = []
            ht_labels = []
            for obj in active_detections[i]:
                ht_boxes.append([obj.rect[0], obj.rect[1],
                                 obj.rect[2] - obj.rect[0],
                                 obj.rect[3] - obj.rect[1]])
                ht_labels.append(obj.label)

            distances = mm.distances.iou_matrix(np.array(gt_boxes),
                                                np.array(ht_boxes), max_iou=0.5)
            accs[i].update(gt_labels, ht_labels, distances)

    log.info('Computing MOT metrics...')
    mh = mm.metrics.create()
    summary = mh.compute_many(accs,
                              metrics=mm.metrics.motchallenge_metrics,
                              generate_overall=True,
                              names=['video ' + str(i) for i in range(len(accs))])

    strsummary = mm.io.render_summary(summary,
                                      formatters=mh.formatters,
                                      namemap=mm.io.motchallenge_metric_names)
    print(strsummary)


if __name__ == '__main__':
    main()
