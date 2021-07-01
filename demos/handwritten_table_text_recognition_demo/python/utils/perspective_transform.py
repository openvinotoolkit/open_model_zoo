import math
import os

import cv2
import numpy as np


def isbadline(p1, p2):
    len2 = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
    if len2 < 400:
        return True
    else:
        return False


def isline(p1, p2, p3):
    if p1[0] == p3[0] and p1[1] == p3[1]:
        return 2
    len13 = math.sqrt((p1[0] - p3[0]) ** 2 + (p1[1] - p3[1]) ** 2)

    if len13 < 5:
        return 2
    len12 = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    len23 = math.sqrt((p2[0] - p3[0]) ** 2 + (p2[1] - p3[1]) ** 2)

    ca = math.fabs(len12 + len23 - len13)
    if ca < 5:
        return 1
    else:
        return 0


# find the crossing point
def computeIntersect(a, b):
    x1, y1, x2, y2 = a[0], a[1], a[2], a[3]
    x3, y3, x4, y4 = b[0], b[1], b[2], b[3]
    d = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))
    if d:
        x = ((x1 * y2 - y1 * x2) * (x3 - x4) -
             (x1 - x2) * (x3 * y4 - y3 * x4)) / d
        y = ((x1 * y2 - y1 * x2) * (y3 - y4) -
             (y1 - y2) * (x3 * y4 - y3 * x4)) / d
        return [x, y]
    else:
        return [-1, -1]


def pstolines(ps):
    lines = []
    for i in range(0, len(ps)):
        j = i + 1
        if j == len(ps):
            j = 0
        if not isbadline(ps[i], ps[j]):
            lines.append([ps[i, 0], ps[i, 1], ps[j, 0], ps[j, 1]])
    return lines


def removebadline(lines):
    for i in range(0, len(lines)):
        j = i + 1
        if j == len(lines):
            j = 0
        if (lines[i][2] != lines[j][0]) or (lines[i][3] != lines[j][1]):
            temp = computeIntersect(lines[i], lines[j])
            lines[i][2] = abs(temp[0])
            lines[j][0] = abs(temp[0])
            lines[i][3] = abs(temp[1])
            lines[j][1] = abs(temp[1])

    return lines


def removemoreline(lines, wise):
    # fclock=False
    while 1:
        linecopy = []
        headp = 0
        for i in range(0, len(lines)):
            j = i + 1
            if j == len(lines):
                j = 0
            temp = isline(lines[i][:2], lines[i][2:], lines[j][2:])
            if temp == 1:
                lines[j][0] = lines[i][0]
                lines[j][1] = lines[i][1]
            elif temp == 2:
                if headp == 0:
                    headp = j
                    linecopy = []
                else:
                    linecopy.append(lines[i])
                    wise[0] = not wise[0]
                    print("wise=", wise)
                    break
            else:
                linecopy.append(lines[i])
        if len(lines) == len(linecopy):
            break
        lines = linecopy
    return lines


def hulllines(lines):
    ppt = []
    for i in range(0, len(lines)):
        ppt.append([lines[i][0], lines[i][1]])
    if lines[0][0] != lines[len(lines) - 1][0] or lines[0][1] != lines[len(lines) - 1][0]:
        ppt.append([lines[len(lines) - 1][0], lines[len(lines) - 1][1]])

    hull = cv2.convexHull(np.array(ppt), clockwise=5)
    hull = hull.reshape(len(hull), 2)
    for i in range(0, len(hull)):
        if hull[i][0] < 0: hull[i][0] = -hull[i][0]
        if hull[i][1] < 0: hull[i][1] = -hull[i][1]
    lines = pstolines(hull)
    lines = removebadline(lines)
    return lines


def drawlines(shape, lines):
    black_canvar = np.zeros(shape, np.uint8)
    for i in range(0, len(lines)):
        cv2.line(black_canvar, (int(lines[i][0]), int(lines[i][1])), (int(lines[i][2]), int(lines[i][3])), 128, 1)
        cv2.circle(black_canvar, (int(lines[i][0]), int(lines[i][1])), 3, 128, 2)
        cv2.putText(black_canvar, str(i), (int(lines[i][0]), int(lines[i][1])),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.4, 255)


def table_tilted_detection(img_path):
    image = cv2.imread(img_path)

    image_copy = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)  # 50,150,3
    lines = cv2.HoughLinesP(edges, 1.0, np.pi / 180, 200, 0, minLineLength=50, maxLineGap=50)  # 650,50,20

    pi = math.pi
    theta_total = 0
    theta_count = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        rho = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        theta = np.math.atan(float(y2 - y1) / float(x2 - x1 + 0.001))
        if pi / 4 > theta > -pi / 4:
            theta_total = theta_total + theta
            theta_count += 1
            cv2.line(image_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)

    theta_average = theta_total / theta_count
    rotate_angle = theta_average * 180 / pi

    return rotate_angle


def perspective_transform(img_path):
    tilted_angle = table_tilted_detection(img_path)

    src_img = cv2.imread(img_path)
    r = 1000.0 / src_img.shape[1]
    dim = (1000, int(src_img.shape[0] * r))
    small_img = cv2.resize(src_img, dim, interpolation=cv2.INTER_AREA)

    if abs(tilted_angle) >= 1:
        img = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
        black = np.zeros(img.shape, np.uint8)
        img = cv2.GaussianBlur(img, (5, 5), 0, 0)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img = cv2.dilate(img, element)
        img = cv2.Canny(img, 20, 40, 3)
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # find the max contour
        max_area = 0
        index = 0
        for i in range(0, len(contours)):
            ps = contours[i].reshape(len(contours[i]), 2)
            r = cv2.boundingRect(ps)
            temp_area = r[2] * r[3]
            if temp_area > max_area:
                index = i
                max_area = temp_area

        ct1 = contours[index]

        # find the four vertex
        contours2 = cv2.approxPolyDP(ct1, 3, True)

        lines = []
        fclock = [False]

        if len(contours2) != 4:
            hull = cv2.convexHull(contours2, clockwise=5)
            c4 = hull.reshape(len(hull), 2)
            lines = pstolines(c4)
            drawlines(img.shape, lines)
            lines = removebadline(lines)
            drawlines(img.shape, lines)

            if len(lines) != 4 and len(lines) != 0:
                lines = removemoreline(lines, fclock)

            if len(lines) != 4 and len(lines) != 0:
                fclock = [False]
                lines = hulllines(lines)
                lines = removemoreline(lines, fclock)

        if len(lines) != 4 and len(lines) != 0:
            for i in range(0, len(lines)):
                cv2.circle(black, (lines[i][0], lines[i][1]), 3, 128, 2)
                cv2.line(black, (lines[i][0], lines[i][1]), (lines[i][2], lines[i][3]), (128), 1)
                cv2.putText(black, str(i), (lines[i][0], lines[i][1]),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.4, 0)

            cv2.circle(black, (lines[len(lines) - 1][0],
                               lines[len(lines) - 1][1]), 3, 128, 2)
            cv2.putText(black, str(len(lines)), (lines[len(lines) - 1][0], lines[len(lines) - 1][1]),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.4, 255)

        if len(lines) == 4:
            pp1 = []
            for i in range(0, 4):
                j = i + 1
                if j == 4:
                    j = 0
                if (lines[i][2] == lines[j][0]) and (lines[i][3] == lines[j][1]):
                    pp1.append([lines[i][2], lines[i][3]])
                else:
                    # find the crossing point
                    temp = computeIntersect(lines[i], lines[j])
                    pp1.append(temp)
            contours2 = np.array(pp1)

        if len(contours2) == 4:
            c4 = contours2.reshape(4, 2)
            if fclock[0]:
                print("c4=", c4)
                c4list = c4.tolist()
                c4list.reverse()
                c4 = np.array(c4list)
                print("Counterclockwise c4=", c4)

            c5 = np.zeros((4, 2), dtype=np.float32)

            minx = 99999
            minindex = 0
            for i in range(0, len(c4)):
                temp = c4[i, 0] ** 2 + c4[i, 1] ** 2
                if minx > temp:
                    minx = temp
                    minindex = i
            for i in range(0, 4):
                c5[i] = c4[minindex]
                minindex = minindex + 1
                if minindex == 4:
                    minindex = 0

            height = math.sqrt((c5[0, 0] - c5[1, 0]) ** 2 + (c5[0, 1] - c5[1, 1]) ** 2)
            height1 = math.sqrt((c5[2, 0] - c5[3, 0]) ** 2 + (c5[2, 1] - c5[3, 1]) ** 2)
            if height < height1:
                height = height1
            width = math.sqrt((c5[0, 0] - c5[3, 0]) ** 2 + (c5[0, 1] - c5[3, 1]) ** 2)
            width1 = math.sqrt((c5[2, 0] - c5[1, 0]) ** 2 + (c5[2, 1] - c5[1, 1]) ** 2)
            if width < width1:
                width = width1

            c2 = np.zeros((4, 2), dtype=np.float32)

            c2[1] = [0, height]
            c2[2] = [width, height]
            c2[3] = [width, 0]
            c2[0] = [0, 0]

            transmtx = cv2.getPerspectiveTransform(c5, c2)
            new_img = np.zeros((np.int0(width), np.int0(height), 3), np.uint8)
            new_img = cv2.warpPerspective(
                small_img, transmtx, (np.int0(width), np.int0(height)), new_img)
            return new_img
    else:
        return small_img
