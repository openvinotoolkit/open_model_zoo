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

from __future__ import print_function

import json
from argparse import ArgumentParser
from collections import namedtuple
from os.path import exists

import json
import numpy as np
from builtins import range
from lxml import etree
from tqdm import tqdm

BBoxDesc = namedtuple('BBoxDesc', 'id, label, det_conf, xmin, ymin, xmax, ymax')
MatchDesc = namedtuple('MatchDesc', 'gt, pred')
Range = namedtuple('Range', 'start, end, label')

UNDEFINED_ACTION_ID = 3
ACTION_NAMES_MAP = {'sitting': 0, 'standing': 1, 'raising_hand': 2, 'listening': 0,
                    'reading': 0, 'writing': 0, 'lie_on_the_desk': 0, 'busy': 0,
                    'in_group_discussions': 0, '__undefined__': UNDEFINED_ACTION_ID}


def load_detections(file_path):
    """Loads dumped detections from the specified file

    :param file_path: Path to file with detections
    :return: Loaded detections
    """

    with open(file_path, 'r') as read_file:
        detections = json.load(read_file)['data']

    out_detections = {}
    for det in tqdm(detections, desc='Extracting detections'):
        frame_id = int(det['frame_id'])
        if frame_id not in out_detections:
            out_detections[frame_id] = []

        xmin, ymin, w, h = det['rect']

        out_detections[frame_id].append(BBoxDesc(id=-1,
                                                 label=int(det['label']),
                                                 det_conf=float(det['det_conf']),
                                                 xmin=float(xmin),
                                                 ymin=float(ymin),
                                                 xmax=float(xmin + w),
                                                 ymax=float(ymin + h)))

    return out_detections


def load_annotation(file_path):
    """Loads annotation from the specified file.

    :param file_path: Path to file with annotation
    :return: Loaded annotation
    """

    tree = etree.parse(file_path)
    root = tree.getroot()

    detections_by_frame_id = {}
    ordered_track_id = -1

    for track in tqdm(root, desc='Extracting annotation'):
        if 'label' not in track.attrib.keys() or track.attrib['label'] != 'person':
            continue

        ordered_track_id += 1
        track_id = ordered_track_id if 'id' not in track.attrib else int(track.attrib['id'])

        for bbox in track:
            if len(bbox) < 1:
                continue

            frame_id = int(bbox.attrib['frame'])
            if frame_id <= 0:
                continue

            action_name = None
            for bbox_attr_id in range(len(bbox)):
                attribute_name = bbox[bbox_attr_id].attrib['name']
                if attribute_name != 'action':
                    continue

                action_name = bbox[bbox_attr_id].text

                break

            if action_name is not None and action_name in ACTION_NAMES_MAP.keys():
                label = ACTION_NAMES_MAP[action_name]

                bbox_desc = BBoxDesc(id=track_id,
                                     label=label,
                                     det_conf=1.0,
                                     xmin=float(bbox.attrib['xtl']),
                                     ymin=float(bbox.attrib['ytl']),
                                     xmax=float(bbox.attrib['xbr']),
                                     ymax=float(bbox.attrib['ybr']))

                detections_by_frame_id[frame_id] = detections_by_frame_id.get(frame_id, []) + [bbox_desc]

    print('Loaded {} annotated frames.'.format(len(detections_by_frame_id)))

    return detections_by_frame_id


def iou(box_a, box_b):
    """ Calculates Intersection over Union (IoU) metric.

    :param box_a: First bbox
    :param box_b: Second bbox
    :return: Scalar value of metric
    """

    intersect_top_left_x = max(box_a.xmin, box_b.xmin)
    intersect_top_left_y = max(box_a.ymin, box_b.ymin)
    intersect_width = max(0.0, min(box_a.xmax, box_b.xmax) - intersect_top_left_x)
    intersect_height = max(0.0, min(box_a.ymax, box_b.ymax) - intersect_top_left_y)

    box_a_area = (box_a.xmax - box_a.xmin) * (box_a.ymax - box_a.ymin)
    box_b_area = (box_b.xmax - box_b.xmin) * (box_b.ymax - box_b.ymin)
    intersect_area = float(intersect_width * intersect_height)

    union_area = float(box_a_area + box_b_area - intersect_area)

    return intersect_area / union_area if union_area > 0.0 else 0.0


def match_detections(predicted_data, gt_data, min_iou):
    """Carry out matching between detected and ground truth bboxes.

    :param predicted_data: List of predicted bboxes
    :param gt_data: List of ground truth bboxes
    :param min_iou: Min IoU value to match bboxes
    :return: List of matches
    """

    all_matches = {}
    total_gt_bbox_num = 0
    matched_gt_bbox_num = 0

    frame_ids = gt_data.keys()
    for frame_id in tqdm(frame_ids, desc='Matching detections'):
        if frame_id not in predicted_data.keys():
            all_matches[frame_id] = []
            continue

        gt_bboxes = gt_data[frame_id]
        predicted_bboxes = predicted_data[frame_id]

        total_gt_bbox_num += len(gt_bboxes)

        sorted_predicted_bboxes = [(i, b) for i, b in enumerate(predicted_bboxes)]
        sorted_predicted_bboxes.sort(key=lambda tup: tup[1].det_conf, reverse=True)

        matches = []
        visited_gt = np.zeros(len(gt_bboxes), dtype=np.bool)
        for i in range(len(sorted_predicted_bboxes)):
            predicted_id = sorted_predicted_bboxes[i][0]
            predicted_bbox = sorted_predicted_bboxes[i][1]

            best_overlap = 0.0
            best_gt_id = -1
            for gt_id in range(len(gt_bboxes)):
                if visited_gt[gt_id]:
                    continue

                overlap_value = iou(predicted_bbox, gt_bboxes[gt_id])
                if overlap_value > best_overlap:
                    best_overlap = overlap_value
                    best_gt_id = gt_id

            if best_gt_id >= 0 and best_overlap > min_iou:
                visited_gt[best_gt_id] = True

                matches.append((best_gt_id, predicted_id))
                matched_gt_bbox_num += 1

                if len(matches) >= len(gt_bboxes):
                    break

        all_matches[frame_id] = matches

    print('Matched gt bbox: {} / {} ({:.2f}%)'
          .format(matched_gt_bbox_num, total_gt_bbox_num,
                  100. * float(matched_gt_bbox_num) / float(max(1, total_gt_bbox_num))))

    return all_matches


def split_to_tracks(gt_data):
    """Splits data to tracks according ID.

    :param gt_data: Input data
    :return: List of tracks
    """

    tracks = {}
    for frame_id in tqdm(gt_data, desc='Splitting GT'):
        gt_frame_data = gt_data[frame_id]
        for bbox in gt_frame_data:
            track_id = bbox.id

            new_match = MatchDesc(bbox, None)

            if track_id not in tracks:
                tracks[track_id] = {frame_id: new_match}
            else:
                tracks[track_id][frame_id] = new_match

    return tracks


def add_matched_predictions(tracks, all_matched_ids, predicted_data, gt_data):
    """Adds matched predicted events to the input tracks.

    :param tracks: Input tracks
    :param all_matched_ids: List of matches
    :param predicted_data: Predicted data
    :param gt_data: ground-truth data
    :return: Updated list of tracks
    """

    for frame_id in tqdm(all_matched_ids.keys(), desc='Splitting Predictions'):
        if len(all_matched_ids[frame_id]) == 0:
            continue

        gt_frame_data = gt_data[frame_id]
        predicted_frame_data = predicted_data[frame_id]
        matched_ids = all_matched_ids[frame_id]

        for match_pair in matched_ids:
            track_id = gt_frame_data[match_pair[0]].id
            predicted_bbox = predicted_frame_data[match_pair[1]]

            new_match = tracks[track_id][frame_id]._replace(pred=predicted_bbox)
            tracks[track_id][frame_id] = new_match

    return tracks


def extract_events(frame_events, window_size, min_length, frame_limits):
    """Merges input frame-based tracks to event-based ones.

    :param frame_events: Input tracks
    :param window_size: Size of smoothing window
    :param min_length: Min duration of event
    :param frame_limits: Start and end frame ID
    :return: List of event-based tracks
    """

    def _smooth(input_events):
        """Merge frames into the events of the same action.

        :param input_events: Frame-based actions
        :return: List of events
        """

        out_events = []

        if len(input_events) > 0:
            last_range = Range(input_events[0][0], input_events[0][0] + 1, input_events[0][1])
            for i in range(1, len(input_events)):
                if last_range.end + window_size - 1 >= input_events[i][0] and last_range.label == input_events[i][1]:
                    last_range = last_range._replace(end=input_events[i][0] + 1)
                else:
                    out_events.append(last_range)

                    last_range = Range(input_events[i][0], input_events[i][0] + 1, input_events[i][1])

            out_events.append(last_range)

        return out_events

    def _filter(input_events):
        """Filters too short events.

        :param input_events: List of events
        :return: Filtered list of events
        """

        return [e for e in input_events if e.end - e.start >= min_length]

    def _extrapolate(input_events):
        """Expands time limits of the input events to the specified one.

        :param input_events: List of events
        :return: Expanded list of events
        """

        out_events = []

        if len(input_events) == 1:
            out_events = [Range(frame_limits[0], frame_limits[1], input_events[0].label)]
        elif len(input_events) > 1:
            first_event = input_events[0]._replace(start=frame_limits[0])
            last_event = input_events[-1]._replace(end=frame_limits[1])
            out_events = [first_event] + input_events[1:-1] + [last_event]

        return out_events

    def _interpolate(input_events):
        """Fills event-free ranges by interpolating neighbouring events.

        :param input_events: List of events
        :return: Filled list of events
        """

        out_events = []

        if len(input_events) > 0:
            last_event = input_events[0]
            for event_id in range(1, len(input_events)):
                cur_event = input_events[event_id]

                middle_point = int(0.5 * (last_event.end + cur_event.start))

                last_event = last_event._replace(end=middle_point)
                cur_event = cur_event._replace(start=middle_point)

                out_events.append(last_event)
                last_event = cur_event
            out_events.append(last_event)

        return out_events

    def _merge(input_events):
        """Merges consecutive events of the same action.

        :param input_events: List of events
        :return: Merged list of events
        """

        out_events = []

        if len(input_events) > 0:
            last_event = input_events[0]
            for cur_event in input_events[1:]:
                if last_event.end == cur_event.start and last_event.label == cur_event.label:
                    last_event = last_event._replace(end=cur_event.end)
                else:
                    out_events.append(last_event)
                    last_event = cur_event
            out_events.append(last_event)

        return out_events

    events = _smooth(frame_events)
    events = _filter(events)
    events = _extrapolate(events)
    events = _interpolate(events)
    events = _merge(events)

    return events


def match_events(gt_events, pred_events):
    """Carry out matching between two input sets of events.

    :param gt_events: Input ground-truth events
    :param pred_events: Input predicted events
    :return: List of matched events
    """

    num_gt = len(gt_events)
    num_pred = len(pred_events)
    if num_gt == 0 or num_pred == 0:
        return []

    matches = []
    for pred_id in range(len(pred_events)):
        best_overlap_value = 0
        best_gt_id = -1
        for gt_id in range(len(gt_events)):
            intersect_start = np.maximum(gt_events[gt_id].start, pred_events[pred_id].start)
            intersect_end = np.minimum(gt_events[gt_id].end, pred_events[pred_id].end)

            overlap = np.maximum(0, intersect_end - intersect_start)
            overlap = 0 if gt_events[gt_id].label != pred_events[pred_id].label else overlap

            if overlap > best_overlap_value:
                best_overlap_value = overlap
                best_gt_id = gt_id

        if best_overlap_value > 0 and best_gt_id >= 0:
            matches.append((best_gt_id, pred_id))

    return matches


def process_tracks(all_tracks, window_size, min_length):
    """Carry out smoothing of the input tracks

    :param all_tracks: Input tracks
    :param window_size: Size of smooth window
    :param min_length: Min duration of event
    :return: List of smoothed tracks
    """

    out_tracks = {}
    for track_id in tqdm(all_tracks.keys(), desc='Extracting events'):
        track = all_tracks[track_id]

        frame_ids = list(track)
        frame_ids.sort()
        frame_id_limits = np.min(frame_ids), np.max(frame_ids) + 1

        gt_frame_events = [(fi, track[fi].gt.label) for fi in frame_ids if track[fi].gt.label != UNDEFINED_ACTION_ID]
        pred_frame_events = [(fi, track[fi].pred.label) for fi in frame_ids if track[fi].pred is not None]

        # skip unmatched track
        if len(gt_frame_events) == 0 or len(pred_frame_events) == 0:
            continue

        gt_events = extract_events(gt_frame_events, window_size, min_length, frame_id_limits)
        pred_events = extract_events(pred_frame_events, window_size, min_length, frame_id_limits)

        out_tracks[track_id] = gt_events, pred_events

    return out_tracks


def calculate_metrics(all_tracks):
    """Calculates Precision and Recall metrics.

    :param all_tracks: Input mathed events
    :return: Precision and Recall scalar values
    """

    total_num_pred_events = 0
    total_num_valid_pred_events = 0
    total_num_gt_events = 0
    total_num_valid_gt_events = 0

    for track_id in all_tracks:
        gt_events, pred_events = all_tracks[track_id]

        matches = match_events(gt_events, pred_events)

        total_num_pred_events += len(pred_events)
        total_num_gt_events += len(gt_events)

        if len(matches) > 0:
            matched_gt = np.zeros([len(gt_events)], dtype=np.bool)
            matched_pred = np.zeros([len(pred_events)], dtype=np.bool)
            for match in matches:
                matched_gt[match[0]] = True
                matched_pred[match[1]] = True

            total_num_valid_pred_events += np.sum(matched_pred)
            total_num_valid_gt_events += np.sum(matched_gt)

    precision = float(total_num_valid_pred_events) / float(total_num_pred_events)\
        if total_num_pred_events > 0 else 0.0
    recall = float(total_num_valid_gt_events) / float(total_num_gt_events)\
        if total_num_gt_events > 0 else 0.0

    return precision, recall


def main():
    """Calculates event-based action metrics.
    """

    parser = ArgumentParser()
    parser.add_argument('--detections', '-d', type=str, required=True, help='Path to .json file with dumped tracks')
    parser.add_argument('--annotation', '-a', type=str, required=True, help='Path to .xml file with annotation')
    parser.add_argument('--min_action_length', type=int, default=30, help='Min action duration (num frames)')
    parser.add_argument('--window_size', type=int, default=30, help='Smooth window size (num frames)')
    args = parser.parse_args()

    assert exists(args.detections)
    assert exists(args.annotation)
    assert args.min_action_length > 0
    assert args.window_size > 0

    detections = load_detections(args.detections)
    annotation = load_annotation(args.annotation)

    all_matches = match_detections(detections, annotation, min_iou=0.5)

    tracks = split_to_tracks(annotation)
    print('Found {} tracks.'.format(len(tracks)))

    tracks = add_matched_predictions(tracks, all_matches, detections, annotation)

    track_events = process_tracks(tracks, args.window_size, args.min_action_length)

    precision, recall = calculate_metrics(track_events)
    print('\nPrecision: {:.3f}%   Recall: {:.3f}%'.format(1e2 * precision, 1e2 * recall))


if __name__ == '__main__':
    main()
