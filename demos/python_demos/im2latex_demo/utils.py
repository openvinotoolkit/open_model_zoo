import pickle as pkl
import json
import numpy as np
import cv2 as cv

COLOR_WHITE = (255, 255, 255)

START_TOKEN = 0
PAD_TOKEN = 1
END_TOKEN = 2
UNK_TOKEN = 3


class Vocab(object):
    def __init__(self, loaded_id2sign):
        assert loaded_id2sign and isinstance(loaded_id2sign, dict)
        self.id2sign = loaded_id2sign

    def construct_phrase(self, indices):
        phrase_converted = []
        for token in indices:
            if token == END_TOKEN:
                break
            phrase_converted.append(
                self.id2sign.get(token, "?"))
        return " ".join(phrase_converted)


def read_vocab(vocab_path):
    if '.pkl' in vocab_path:
        with open(vocab_path, "rb") as f:
            vocab_dict = pkl.load(f)
    elif 'json' in vocab_path:
        with open(vocab_path, "r") as f:
            vocab_dict = json.load(f)
            for k, v in vocab_dict['id2sign'].items():
                del vocab_dict['id2sign'][k]
                vocab_dict['id2sign'][int(k)] = v
    else:
        raise ValueError("Wrong extension of the vocab file")
    vocab = Vocab(loaded_id2sign=vocab_dict["id2sign"])
    return vocab


class ResizePadToTGTShape():
    def __init__(self, target_shape):
        self.target_height, self.target_width = target_shape

    def __call__(self, image_raw):

        img_h, img_w = image_raw.shape[0:2]
        if (img_h, img_w) != (self.target_height, self.target_width):
            if img_h >= self.target_height and img_w >= self.target_width:
                rescale_h = img_h / self.target_height
                rescale_w = img_w / self.target_width
                if rescale_h > rescale_w:
                    new_h = int(img_h / rescale_h)
                    new_w = int(img_w / rescale_h)
                else:
                    new_h = int(img_h / rescale_w)
                    new_w = int(img_w / rescale_w)

                image_raw = cv.resize(image_raw, (new_w, new_h))
                img_h, img_w = image_raw.shape[0:2]
                if (img_h, img_w != self.target_height, self.target_width):
                    image_raw = cv.copyMakeBorder(image_raw, 0, self.target_height - img_h,
                                                  0, self.target_width - img_w, cv.BORDER_CONSTANT,
                                                  None, COLOR_WHITE)
            elif img_h < self.target_height and img_w < self.target_width:
                rescale_h = img_h / self.target_height
                rescale_w = img_w / self.target_width
                if rescale_h > rescale_w:
                    new_h = int(img_h / rescale_h)
                    new_w = int(img_w / rescale_h)
                else:
                    new_h = int(img_h / rescale_w)
                    new_w = int(img_w / rescale_w)
                image_raw = cv.resize(image_raw, (new_w, new_h))
                img_h, img_w = image_raw.shape[0:2]
                if (img_h, img_w != self.target_height, self.target_width):
                    image_raw = cv.copyMakeBorder(image_raw, 0, self.target_height - img_h,
                                                  0, self.target_width - img_w, cv.BORDER_CONSTANT,
                                                  None, COLOR_WHITE)
            elif img_h < self.target_height and img_w >= self.target_width:
                dim = (self.target_width, int(
                    self.target_width * img_h / img_w))
                image_raw = cv.resize(image_raw, dim)
                image_raw = cv.copyMakeBorder(image_raw, 0, self.target_height - image_raw.shape[0],
                                              0, 0, cv.BORDER_CONSTANT, None,
                                              COLOR_WHITE)
            elif img_h >= self.target_height and img_w < self.target_width:
                dim = (int(self.target_height * img_w / img_h),
                       self.target_height)
                image_raw = cv.resize(image_raw, dim)
                img_h, img_w = image_raw.shape[0:2]
                image_raw = cv.copyMakeBorder(image_raw, 0, 0,
                                              0, self.target_width - img_w, cv.BORDER_CONSTANT,
                                              None, COLOR_WHITE)
        return image_raw


class CropPadToTGTShape():
    def __init__(self, target_shape):
        self.target_height, self.target_width = target_shape

    def __call__(self, image_raw):
        img_h, img_w = image_raw.shape[0:2]
        if (img_h, img_w) != (self.target_height, self.target_width):
            if img_h >= self.target_height and img_w >= self.target_width:
                if len(image_raw.shape) > 2:
                    image_raw = image_raw[:self.target_height,
                                          :self.target_width, :]
                    assert image_raw.shape[1] == self.target_width
                else:
                    image_raw = image_raw[:self.target_height,
                                          :self.target_width]
            elif img_h < self.target_height and img_w < self.target_width:

                image_raw = cv.copyMakeBorder(image_raw, 0, self.target_height - img_h,
                                              0, self.target_width - img_w, cv.BORDER_CONSTANT,
                                              None, COLOR_WHITE)
            elif img_h < self.target_height and img_w >= self.target_width:
                if len(image_raw.shape) > 2:
                    image_raw = image_raw[:, :self.target_width, :]
                else:
                    image_raw = image_raw[:, :self.target_width]
                image_raw = cv.copyMakeBorder(image_raw, 0, self.target_height - image_raw.shape[0],
                                              0, 0, cv.BORDER_CONSTANT, None,
                                              COLOR_WHITE)
            elif img_h >= self.target_height and img_w < self.target_width:
                if len(image_raw.shape) > 2:
                    image_raw = image_raw[:self.target_height, :, :]
                else:
                    image_raw = image_raw[:self.target_height, :]
                img_h, img_w = image_raw.shape[0:2]
                image_raw = cv.copyMakeBorder(image_raw, 0, 0,
                                              0, self.target_width - img_w, cv.BORDER_CONSTANT,
                                              None, COLOR_WHITE)
        return image_raw
