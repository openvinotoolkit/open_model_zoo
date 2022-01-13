import os

import numpy as np
from src.evaluators.pascal_voc_evaluator import (calculate_ap_11_point_interp,
                                                 calculate_ap_every_point)
from src.tube import Tube
from src.utils.enumerators import MethodAveragePrecision
from src.utils.read_files import File


class TubeEvaluator():
    """ Spatio-temporal tube evaluator
    """
    def __init__(self,
                 anno_filepath,
                 preds_filepath,
                 method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION):
        """Class constructor

        Args:
            anno_filepath (str): annotation filepath
            preds_filepath (str): prediction filepath in json extension
            method (MethodAveragePrecision, optional): Recall interpolation method (see src.utils.enumerators). Defaults to MethodAveragePrecision.EVERY_POINT_INTERPOLATION.
        """

        if not anno_filepath.endswith('.json'):
            raise ValueError("Invalid extension file: ", anno_filepath)
        if not preds_filepath.endswith('.json'):
            raise ValueError("Invalid extension file:", preds_filepath)

        self._anno_filepath = anno_filepath
        self._preds_filepath = preds_filepath
        self._method = method

    def __reset(self):
        """reset object
        """
        self._gt = []
        self._predictions = []
        self.nb_tp = 0
        self.nb_fp = 0
        self.nb_fn = 0
        self.nb_tn = -1
        self._res = dict()
        # self._video_id = 0

    def __process(self):
        """process the files and prepare to evaluate
        """
        self.__reset()

        annot_data = File(self._anno_filepath).read()
        pred_data = File(self._preds_filepath).read()

        # List with all gts tubes (Ex: [Tube(num_instances=5...99 0.98]]), Tube(num_instances=5...99 0.89]])] )
        self._gt = [Tube(**annot) for annot in annot_data['annotations']]
        # List with all preds tubes
        self._predictions = [Tube(**pred) for pred in pred_data]

        self._videos = annot_data['videos']
        self._classes = annot_data['categories']

    def evaluate(self, thr=0.5):
        """Evaluate the predictions according to the chosen IOU threshold

        Args:
            thr (float, optional): IOU threshold 0 < thr < 1. Defaults to 0.5.

        Returns:
            res, mAP: return a dictionary (res) with results per class. Also, returns the mAP.
        """
        if not 0 < thr <= 1:
            raise ValueError("IOU threshold must be 0 < thr <= 1: ", thr)

        self.__process()

        # loop over classes
        # TODO: group detection on videos and classes basis to avoid loop multiple times
        for obj_cls in self._classes:
            gt_cls = [gt for gt in self._gt if gt.category_id == obj_cls['id']]
            preds_cls = [pred for pred in self._predictions if pred.category_id == obj_cls['id']]

            # sort detections by decreasing confidence
            preds_cls = sorted(preds_cls, key=lambda tube: tube.confidence, reverse=True)

            # initialize true positive as zeros of length of predictions of a giving class

            # loop over videos
            for vid_id in self._videos:
                gts = [gt for gt in gt_cls if gt.video_id == vid_id['id']]
                preds = [pred for pred in preds_cls if pred.video_id == vid_id['id']]

                n_tp, n_fp, n_fn = self._classify_tubes(preds, gts, thr)

            TP = np.array([int(tube.isTP) for tube in preds_cls])
            FP = np.logical_not(TP).astype(int)

            # compute precision, recall and average precision
            acc_TP = np.cumsum(TP)
            acc_FP = np.cumsum(FP)
            rec = acc_TP / len(gt_cls)
            prec = np.divide(acc_TP, (acc_FP + acc_TP))

            # Depending on the method, call the right implementation
            if self._method == MethodAveragePrecision.EVERY_POINT_INTERPOLATION:
                [ap, mpre, mrec, ii] = calculate_ap_every_point(rec, prec)
            elif self._method == MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION:
                [ap, mpre, mrec, _] = calculate_ap_11_point_interp(rec, prec)
            else:
                raise ValueError(f'Invalid interpolation method: {self._method}')

            # add class result in the dictionary to be returned
            self._res[obj_cls['name']] = {
                'precision': prec,
                'recall': rec,
                'AP': ap,
                'interpolated precision': mpre,
                'interpolated recall': mrec,
                'total TP': n_tp,
                'total FP': n_fp,
                'total FN': n_fn,
            }
        # For mAP, only the classes in the gt set should be considered
        mAP = 0.0
        for c, r in self._res.items():
            if any(cat['name'] == c for cat in self._classes):
                mAP += r['AP']
        mAP /= len(self._classes)

        return self._res, mAP

    def _classify_tubes(self, preds: list, gts: list, thr: float) -> tuple:
        """This method classify the `preds` in TP or !TP and the `gts` in FN or !FN, by setting an attribute in each tube in the lists.
        This is done according to the threshold chosen. Detections with higher confidences have priority.

        Args:
            preds (list): list of  predicted Tube objects
            gts (list): list of  annotation Tube objects
            thr (float): threshold to consider a tube correctly detected. It compares the Spatio-temporal IOU (STT-IOU).

        Returns:
            tuple: return the number of TP, FP and FN.
        """

        gt_overlaps = np.zeros(len(gts))
        overlaps = self._tube_pairwise_iou(preds, gts)

        # consider no detections at first
        [gt.__setattr__('isFN', True) for gt in gts]
        [pred.__setattr__('isTP', False) for pred in preds]

        for j in range(min(len(preds), len(gts))):
            max_overlaps = overlaps.max(axis=0)
            argmax_overlaps = overlaps.argmax(axis=0)

            # find which gt tube is 'best' covered (i.e. 'best' = most iou)
            gt_ovr = max_overlaps.max(axis=0)
            gt_ind = max_overlaps.argmax(axis=0)
            assert gt_ovr >= 0

            # find the proposal tube that covers the best covered gt tube
            box_ind = argmax_overlaps[gt_ind]

            # in case the ovr greater than threshold,
            # set the gt tube to detected and pred as TP
            if gt_ovr >= thr:
                gts[gt_ind].__setattr__('isFN', False)
                preds[box_ind].__setattr__('isTP', True)

            # record the iou coverage of this gt tube
            gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert gt_overlaps[j] == gt_ovr

            # mark the proposal tube and the gt tube as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        nb_tp = (gt_overlaps >= thr).astype(int).sum()
        assert nb_tp >= 0

        nb_fp = len(preds) - nb_tp
        assert nb_fp >= 0

        nb_fn = len(gts) - nb_tp
        assert nb_fn >= 0

        return nb_tp, nb_fp, nb_fn

    def _tube_pairwise_iou(self, preds: list, gts: list) -> np.array:
        """compute the pairwise spatio-temporal tube iou (STT-IOU)

        Args:
            preds (list): list of tube predictions
            gts (list): list of tube annotation

        Returns:
            np.array: pairwise STT-IOU
        """
        # initialize matrices that will keep the intersection and union of the tubes
        inter = np.zeros((len(preds), len(gts)))
        union = np.zeros((len(preds), len(gts)))

        pred_idx = 0
        for pred in preds:
            gt_idx = 0

            for gt in gts:
                # compute the pairwise intersection
                inter[pred_idx, gt_idx] = self._tubes_inter(pred, gt)
                # compute the pairwise union
                union[pred_idx, gt_idx] = self._tubes_union(pred, gt)

                gt_idx += 1

            pred_idx += 1

        iou = inter / (union - inter)

        return iou

    def _tubes_inter(self, tube1: Tube, tube2: Tube) -> np.array:
        """Give two tubes of track size N and M,
        compute de intersection volume between __all__ N x M pairs of boxes.
        The boxes in track must be (xmin, ymin, xmax, ymax)

        Args:
            tube1, tube2 (Tube): two `Tubes`. Contains tracks with N & M boxes, respectively.

        Returns:
            np.array: intersection volume.
        """

        inter_frames = self._get_intersection_frames(tube1, tube2)

        inter = 0

        if inter_frames is not None:
            for inter_frame in inter_frames:
                box1 = tube1.get_frame_boxes(inter_frame)
                box2 = tube2.get_frame_boxes(inter_frame)

                width_height = np.minimum(box1[:, None, 2:], box2[:, 2:]) - np.maximum(
                    box1[:, None, :2], box2[:, :2])  # [N,M,2]

                width_height.clip(min=0, out=width_height)  # [N,M,2]
                inter += width_height.prod(axis=2)  # [N,M]
                del width_height

        return inter

    def _tubes_union(self, tube1: Tube, tube2: Tube) -> np.array:
        """Give two tubes of track size N and M,
        compute de union volume between __all__ N x M pairs of boxes.

        Args:
            tube1, tube2 (Tube): two `Tubes`. Contains tracks with N & M boxes, respectively.

        Returns:
            np.array: union volume.
        """
        vol1 = tube1.get_tube_volume()
        vol2 = tube2.get_tube_volume()
        return vol1 + vol2

    def _get_intersection_frames(self, tube1: Tube, tube2: Tube) -> list:
        """Get the frames that are commom to tube1 and tube2.

        Args:
            tube1, tube2 (Tube): two `Tubes`. Contains tracks with N & M boxes, respectively.

        Returns:
            list: list of frames commom to tube1 and tube2.
        """
        frames_tube1 = set(tube1.get_frames())
        frames_tube2 = set(tube2.get_frames())

        frames_inter = frames_tube1.intersection(frames_tube2)

        if len(frames_inter) > 0:
            return frames_inter

        return None
