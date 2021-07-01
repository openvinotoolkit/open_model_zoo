import os
import shutil

import cv2
import numpy as np

from utils.perspective_transform import perspective_transform


def isolate(img):
    idx = np.argwhere(img < 1)
    rows, cols = img.shape

    for i in range(idx.shape[0]):
        c_row = idx[i, 0]
        c_col = idx[i, 1]
        if c_col + 1 < cols and c_row + 1 < rows:
            img[c_row, c_col + 1] = 1
            img[c_row + 1, c_col] = 1
            img[c_row + 1, c_col + 1] = 1
        if c_col + 2 < cols and c_row + 2 < rows:
            img[c_row + 1, c_col + 2] = 1
            img[c_row + 2, c_col] = 1
            img[c_row, c_col + 2] = 1
            img[c_row + 2, c_col + 1] = 1
            img[c_row + 2, c_col + 2] = 1
    return img


def clearEdge(img, width):
    img[0:width - 1, :] = 1
    img[1 - width:-1, :] = 1
    img[:, 0:width - 1] = 1
    img[:, 1 - width:-1] = 1
    return img


def point2line_distance(x1, y1, x2, y2, pointPx, pointPy):
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - y1 * x2
    distance = abs(A * pointPx + B * pointPy + C) / ((A * A + B * B) ** 0.5)
    return distance


def table_test(image):
    cells_path = os.path.join(os.path.dirname(__file__), '../cells') 
    image_copy = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)
    rows, cols = binary.shape

    scale = 25
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilatedcol = cv2.dilate(eroded, kernel, iterations=1)

    scale = 110
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // scale))
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilatedrow = cv2.dilate(eroded, kernel, iterations=1)

    merge = cv2.add(dilatedcol, dilatedrow)

    merge_h, merge_w = merge.shape
    fix_h, fix_w = int(merge_h * 0.03), int(merge_w * 0.03)

    merge[0:fix_h, 0:merge_w] = 1
    merge[merge_h - fix_h:merge_h, 0:merge_w] = 1
    merge[0:merge_h, 0:fix_w] = 1
    merge[0:merge_h, merge_w - fix_w:merge_w] = 1

    contours, hierarchy = cv2.findContours(merge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    length = len(contours)

    rec_num = 0

    temp_pre = 0, 0, 0, 0

    detc_results = dict()
    for i in range(length):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area < 100:
            continue

        approx = cv2.approxPolyDP(cnt, 3, True)  # 3

        x, y, w, h = cv2.boundingRect(approx)

        # avoid the same and approximate contours
        if abs(x - temp_pre[0]) <= 10 and abs(y - temp_pre[1]) <= 10:
            continue
        temp_pre = x, y, w, h

        if 10 < h < 600 and w > 10:  # important 10 < h < 600 and w > 10
            rec_roi = image_copy[y:y + h, x:x + w]
            rec_num = rec_num + 1
            save_path = '{0:0>4}_{1:0>6}'.format(1, i)

            detc_results[save_path] = (x, y, w, h)

    for key in detc_results:
        x, y, w, h = detc_results.get(key)
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (255 - h * 3, h * 3, 0), 3)

    print('rec_num: ', rec_num)
        
    return detc_results, image_copy

