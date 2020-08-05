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
    def __init__(self, loaded_sign2id=None, loaded_id2sign=None):
        if loaded_id2sign is None and loaded_sign2id is None:
            self.sign2id = {"<s>": START_TOKEN, "</s>": END_TOKEN,
                            "<pad>": PAD_TOKEN, "<unk>": UNK_TOKEN}
            self.id2sign = dict((idx, token)
                                for token, idx in self.sign2id.items())
            self.length = 4
        else:
            assert isinstance(loaded_id2sign, dict) and isinstance(
                loaded_sign2id, dict)
            assert len(loaded_id2sign) == len(loaded_sign2id)
            self.sign2id = loaded_sign2id
            self.id2sign = loaded_id2sign
            self.length = len(loaded_id2sign)

    def __len__(self):
        return self.length

    def construct_phrase(self, indices, max_len=None, skip_end_token=True):
        phrase_converted = []
        if max_len is not None:
            indices_to_convert = indices[:max_len]
        else:
            indices_to_convert = indices

        for token in indices_to_convert:
            val = token.item()
            if val == END_TOKEN and skip_end_token:
                break
            phrase_converted.append(
                self.id2sign.get(val, "?"))
            if val == END_TOKEN:
                break

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
    vocab = Vocab(
        loaded_id2sign=vocab_dict["id2sign"], loaded_sign2id=vocab_dict["sign2id"])
    return vocab


class BatchResizePadToTGTShape():
    def __init__(self, target_shape):
        self.target_shape = target_shape

    def __call__(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        res = []
        target_height, target_width = self.target_shape
        for image_raw in imgs:

            img_h, img_w = image_raw.shape[0:2]
            if (img_h, img_w) != (target_height, target_width):
                if img_h >= target_height and img_w >= target_width:
                    rescale_h = img_h / target_height
                    rescale_w = img_w / target_width
                    if rescale_h > rescale_w:
                        new_h = int(img_h / rescale_h)
                        new_w = int(img_w / rescale_h)
                    else:
                        new_h = int(img_h / rescale_w)
                        new_w = int(img_w / rescale_w)

                    image_raw = cv.resize(image_raw, (new_w, new_h))
                    img_h, img_w = image_raw.shape[0:2]
                    if (img_h, img_w != target_height, target_width):
                        image_raw = cv.copyMakeBorder(image_raw, 0, target_height - img_h,
                                                      0, target_width - img_w, cv.BORDER_CONSTANT,
                                                      None, COLOR_WHITE)
                elif img_h < target_height and img_w < target_width:
                    rescale_h = img_h / target_height
                    rescale_w = img_w / target_width
                    if rescale_h > rescale_w:
                        new_h = int(img_h / rescale_h)
                        new_w = int(img_w / rescale_h)
                    else:
                        new_h = int(img_h / rescale_w)
                        new_w = int(img_w / rescale_w)
                    image_raw = cv.resize(image_raw, (new_w, new_h))
                    img_h, img_w = image_raw.shape[0:2]
                    if (img_h, img_w != target_height, target_width):
                        image_raw = cv.copyMakeBorder(image_raw, 0, target_height - img_h,
                                                      0, target_width - img_w, cv.BORDER_CONSTANT,
                                                      None, COLOR_WHITE)
                elif img_h < target_height and img_w >= target_width:
                    dim = (target_width, int(target_width * img_h / img_w))
                    image_raw = cv.resize(image_raw, dim)
                    image_raw = cv.copyMakeBorder(image_raw, 0, target_height - image_raw.shape[0],
                                                  0, 0, cv.BORDER_CONSTANT, None,
                                                  COLOR_WHITE)
                elif img_h >= target_height and img_w < target_width:
                    dim = (int(target_height * img_w / img_h), target_height)
                    image_raw = cv.resize(image_raw, dim)
                    img_h, img_w = image_raw.shape[0:2]
                    image_raw = cv.copyMakeBorder(image_raw, 0, 0,
                                                  0, target_width - img_w, cv.BORDER_CONSTANT,
                                                  None, COLOR_WHITE)
            res.append(image_raw)
        return res


class BatchCropPadToTGTShape():
    def __init__(self, target_shape):
        self.target_shape = target_shape

    def __call__(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        res = []
        target_height, target_width = self.target_shape
        for image_raw in imgs:

            img_h, img_w = image_raw.shape[0:2]
            if (img_h, img_w) != (target_height, target_width):
                if img_h >= target_height and img_w >= target_width:
                    if len(image_raw.shape) > 2:
                        image_raw = image_raw[:target_height, :target_width, :]
                        assert image_raw.shape[1] == target_width
                    else:
                        image_raw = image_raw[:target_height, :target_width]
                elif img_h < target_height and img_w < target_width:

                    image_raw = cv.copyMakeBorder(image_raw, 0, target_height - img_h,
                                                  0, target_width - img_w, cv.BORDER_CONSTANT,
                                                  None, COLOR_WHITE)
                elif img_h < target_height and img_w >= target_width:
                    if len(image_raw.shape) > 2:
                        image_raw = image_raw[:, :target_width, :]
                    else:
                        image_raw = image_raw[:, :target_width]
                    image_raw = cv.copyMakeBorder(image_raw, 0, target_height - image_raw.shape[0],
                                                  0, 0, cv.BORDER_CONSTANT, None,
                                                  COLOR_WHITE)
                elif img_h >= target_height and img_w < target_width:
                    if len(image_raw.shape) > 2:
                        image_raw = image_raw[:target_height, :, :]
                    else:
                        image_raw = image_raw[:target_height, :]
                    img_h, img_w = image_raw.shape[0:2]
                    image_raw = cv.copyMakeBorder(image_raw, 0, 0,
                                                  0, target_width - img_w, cv.BORDER_CONSTANT,
                                                  None, COLOR_WHITE)
            res.append(image_raw)
        return res
