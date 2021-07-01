import os
import sys
import time
import logging as log

from tqdm import tqdm
from PySide2.QtCore import QThread, Signal
from openvino.inference_engine import IECore
import cv2
import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk

from utils.codec import CTCCodec


class OcrWorker(QThread):
    ocr_finished = Signal(dict)

    def __init__(self, args):
        super().__init__()
        self.detc_results = None

        log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
        # Plugin initialization
        ie = IECore()
        # Read IR
        log.info("Loading network")
        net = ie.read_network(args.model,
                              os.path.splitext(args.model)[0] + ".bin")

        assert len(net.input_info) == 1, "Demo supports only single input topologies"
        assert len(net.outputs) == 1, "Demo supports only single output topologies"

        log.info("Preparing input/output blobs")
        self.input_blob = next(iter(net.input_info.keys()))
        self.out_blob = next(iter(net.outputs.keys()))

        with open(args.charlist, 'r', encoding='utf-8') as f:
            characters = ''.join(line.strip('\n') for line in f)

        self.codec = CTCCodec(characters, args.designated_characters, args.top_k)
        assert len(self.codec.characters) == net.outputs[self.out_blob].shape[
            2], "The text recognition model does not correspond to decoding character list"

        self.input_batch_size, self.input_channel, self.input_height, self.input_width = net.inputs[
            self.input_blob].shape

        # Loading model to the plugin
        log.info("Loading model to the plugin")
        self.exec_net = ie.load_network(network=net, device_name=args.device)

    def get_detc_results(self, detc_results):
        self.detc_results = detc_results

    def get_img(self, img):
        self.img = img

    def run(self):
        log.info('Handwritten text recognition begins')

        results = {}
        infer_time = []
        for name, xywh in tqdm(self.detc_results.items()):
            x, y, w, h = xywh
            t0 = time.time()
            # read and pre_process input image(note: one image only)
            img = self.img[y:y+h,x:x+w, :]
            input_image = self.pre_process_input(img, height=self.input_height, width=self.input_width)[None, :, :, :]
            assert self.input_batch_size == input_image.shape[
                0], "the net's batch size should equal the input image's batch size"
            assert self.input_channel == input_image.shape[
                1], "the net's input channel should equal the input image's channel"
            preds = self.exec_net.infer(inputs={self.input_blob: input_image})
            preds = preds[self.out_blob]
            result = self.codec.decode(preds)
            results[name] = result[0]
            infer_time.append((time.time() - t0) * 1000)
        log.info('Handwritten text recognition is done')
        log.info('Average throughout: {} ms'.format(np.average(np.asarray(infer_time))))
        log.info('Emit the results')
        self.ocr_finished.emit(results)

    def pre_process_input(self, img, height, width):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # calculate local entropy
        entr = entropy(img, disk(5))
        # Normalize and negate entropy values
        MAX_ENTROPY = 8.0
        MAX_PIX_VAL = 255
        negative = 1 - (entr / MAX_ENTROPY)
        u8img = (negative * MAX_PIX_VAL).astype(np.uint8)
        # Global thresholding
        ret, mask = cv2.threshold(u8img, 0, MAX_PIX_VAL, cv2.THRESH_OTSU)
        # mask out text
        masked = cv2.bitwise_and(img, img, mask=mask)
        # fill in the holes to estimate the background
        kernel = np.ones((35, 35), np.uint8)
        background = cv2.dilate(masked, kernel, iterations=1)
        # By subtracting background from the original image, we get a clean text image
        text_only = cv2.absdiff(img, background)
        # Negate and increase contrast
        neg_text_only = (MAX_PIX_VAL - text_only) * 1.15
        # clamp the image within u8 range
        ret, clamped = cv2.threshold(neg_text_only, 255, MAX_PIX_VAL, cv2.THRESH_TRUNC)
        clamped_u8 = clamped.astype(np.uint8)
        # Do final adaptive thresholding to binarize image
        processed = cv2.adaptiveThreshold(clamped_u8, MAX_PIX_VAL, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                          31, 2)
        ratio = float(processed.shape[1]) / float(processed.shape[0])
        tw = int(height * ratio)
        rsz = cv2.resize(processed, (tw, height), interpolation=cv2.INTER_AREA).astype(np.float32)

        # [h,w] -> [c,h,w]
        img = rsz[None, :, :]
        _, h, w = img.shape
        # right edge padding
        pad_img = np.pad(img, ((0, 0), (0, height - h), (0, width - w)), mode='edge')

        return pad_img
