#!/usr/bin/env python3

import json
import glob
import os
from pathlib import Path
import numpy as np
import cv2
from skimage.filters.rank import entropy
from skimage.morphology import disk

TEST_FOLDER="test/"
TEST_TXT=TEST_FOLDER+"test_img_id_gt.txt"

def binarize(img):
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
    return processed

def main():
    """
    Preprocess GNHK dataset into text line images
    """
    os.makedirs(os.path.dirname(TEST_FOLDER), exist_ok=True)
    open(TEST_TXT, 'w').close() # clear the file
    images = glob.glob('eng*.jpg')

    for img_idx, image in enumerate(images):
        img_id = Path(image).stem
        print(img_idx, img_id)
        # open corresponding JSON annotation file
        with open(img_id + ".json") as f:
            data = json.load(f)
            line_indices = {obj["line_idx"] for obj in data}
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            img = binarize(img)
            for idx in sorted(line_indices):
                objects = list(filter(lambda obj: obj["line_idx"] == idx, data))
                # discard math symbols, scribbles, illegible text, and printed text
                objects = list(filter(lambda obj: obj["text"] != "%math%" and obj["text"] != "%SC%" and obj["text"] != "%NA%" and obj["type"] != "P", objects))
                if not objects:
                    continue
                objects = sorted(objects, key=lambda x: x['polygon']['x0'])
                label = " ".join((obj["text"] for obj in objects))
                print(img_id, idx, label)

                # create mask for the words
                mask = np.zeros(img.shape[0:2], dtype=np.uint8)
                for obj in objects:
                    region = [
                        [obj["polygon"]["x0"], obj["polygon"]["y0"]],
                        [obj["polygon"]["x1"], obj["polygon"]["y1"]],
                        [obj["polygon"]["x2"], obj["polygon"]["y2"]],
                        [obj["polygon"]["x3"], obj["polygon"]["y3"]]
                    ]
                    points = np.array([region])
                    cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
                masked = cv2.bitwise_and(img, img, mask = mask)
                bg = np.ones_like(img, np.uint8) * 255
                cv2.bitwise_not(bg, bg, mask = mask)
                overlay = bg + masked
                # crop bounding rectangle of the text region
                polys = [[
                    [obj["polygon"]["x0"], obj["polygon"]["y0"]],
                    [obj["polygon"]["x1"], obj["polygon"]["y1"]],
                    [obj["polygon"]["x2"], obj["polygon"]["y2"]],
                    [obj["polygon"]["x3"], obj["polygon"]["y3"]]
                ] for obj in objects]
                flat = [item for sublist in polys for item in sublist]
                pts = np.array(flat)
                rect = cv2.boundingRect(pts)
                x, y, w, h = rect
                cropped = overlay[y:y+h, x:x+w].copy()

                # discard image if width > 2000 after resizing to height=96 while keeping aspect ratio
                height, width = cropped.shape
                ratio = 96.0 / height
                new_width = int(width * ratio)
                if new_width > 2000:
                    continue

                cv2.imwrite(TEST_FOLDER + img_id + '_line'+ str(idx) + '.jpg', cropped)
                with open(TEST_TXT, 'a', encoding='utf-8') as test_txt:
                    test_txt.write(img_id + '_line'+ str(idx) + '.jpg' + ',' + label + '\n')

if __name__ == '__main__':
    main()
