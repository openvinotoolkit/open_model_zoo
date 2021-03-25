import numpy as np


def rect2square(rectangles):
    """
    Function:
        change rectangles into squares (matrix version)
    Input:
        rectangles: rectangles[i][0:3] is the position, rectangles[i][4] is score
    Output:
        squares: same as input
    """
    w = rectangles[:, 2] - rectangles[:, 0]
    h = rectangles[:, 3] - rectangles[:, 1]
    L = np.maximum(w, h).T
    rectangles[:, 0] = rectangles[:, 0] + w*0.5 - L*0.5
    rectangles[:, 1] = rectangles[:, 1] + h*0.5 - L*0.5
    rectangles[:, 2:4] = rectangles[:, 0:2] + np.repeat([L], 2, axis=0).T
    return rectangles


def NMS(rectangles, threshold, use_iom=False):
    """
    Function:
        apply NMS(non-maximum suppression) on ROIs in same scale(matrix version)
    Input:
        rectangles: rectangles[i][0:3] is the position, rectangles[i][4] is score
    Output:
        rectangles: same as input
    """
    if len(rectangles) == 0:
        return rectangles
    boxes = np.array(rectangles)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s  = boxes[:, 4]
    area = np.multiply(x2-x1+1, y2-y1+1)
    i = np.array(s.argsort())
    pick = []
    while len(i) > 0:
        xx1 = np.maximum(x1[i[-1]], x1[i[0:-1]]) #I[-1] have hightest prob score, I[0:-1]->others
        yy1 = np.maximum(y1[i[-1]], y1[i[0:-1]])
        xx2 = np.minimum(x2[i[-1]], x2[i[0:-1]])
        yy2 = np.minimum(y2[i[-1]], y2[i[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if use_iom:
            o = inter / np.minimum(area[i[-1]], area[i[0:-1]])
        else:
            o = inter / (area[i[-1]] + area[i[0:-1]] - inter)
        pick.append(i[-1])
        i = i[np.where(o <= threshold)[0]]
    result_rectangle = boxes[pick].tolist()
    return result_rectangle


def detect_face_12net(cls_prob, roi, out_side, scale, width, height, score_threshold, iou_threshold):
    """
    Function:
        Detect face position and calibrate bounding box on 12net feature map(matrix version)
    Input:
        cls_prob : softmax feature map for face classify
        roi      : feature map for regression
        out_side : feature map's largest size
        scale    : current input image scale in multi-scales
        width    : image's origin width
        height   : image's origin height
        threshold: 0.6 can have 99% recall rate
    """
    in_side = 2*out_side+11
    stride = 0
    if out_side != 1:
        stride = float(in_side-12)/(out_side-1)
    (x, y) = np.where(cls_prob >= score_threshold)
    boundingbox = np.array([x, y]).T
    bb1 = np.fix((stride * (boundingbox) + 0)  * scale)
    bb2 = np.fix((stride * (boundingbox) + 11) * scale)
    boundingbox = np.concatenate((bb1, bb2), axis=1)
    dx1 = roi[0][x, y]
    dx2 = roi[1][x, y]
    dx3 = roi[2][x, y]
    dx4 = roi[3][x, y]
    score = np.array([cls_prob[x, y]]).T
    offset = np.array([dx1, dx2, dx3, dx4]).T
    boundingbox = boundingbox + offset*12.0*scale
    rectangles = np.concatenate((boundingbox, score), axis=1)
    rectangles = rect2square(rectangles)
    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0,      rectangles[i][0]))
        y1 = int(max(0,      rectangles[i][1]))
        x2 = int(min(width,  rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        sc = rectangles[i][4]
        if x2 > x1 and y2 > y1:
            pick.append([x1, y1, x2, y2, sc])
    return NMS(pick, iou_threshold)


def filter_face_24net(cls_prob, roi, rectangles, width, height, score_threshold, iou_threshold):
    """
    Function:
        Filter face position and calibrate bounding box on 12net's output
    Input:
        cls_prob  : softmax feature map for face classify
        roi_prob  : feature map for regression
        rectangles: 12net's predict
        width     : image's origin width
        height    : image's origin height
        threshold : 0.6 can have 97% recall rate
    Output:
        rectangles: possible face positions
    """
    prob = cls_prob[:, 1]
    pick = np.where(prob >= score_threshold)
    rectangles = np.array(rectangles)
    x1 = rectangles[pick, 0]
    y1 = rectangles[pick, 1]
    x2 = rectangles[pick, 2]
    y2 = rectangles[pick, 3]
    sc = np.array([prob[pick]]).T

    dx1 = roi[pick, 0]
    dx2 = roi[pick, 1]
    dx3 = roi[pick, 2]
    dx4 = roi[pick, 3]

    w = x2 - x1
    h = y2 - y1

    x1 = np.array([(x1+dx1*w)[0]]).T
    y1 = np.array([(y1+dx2*h)[0]]).T
    x2 = np.array([(x2+dx3*w)[0]]).T
    y2 = np.array([(y2+dx4*h)[0]]).T

    rectangles = np.concatenate((x1, y1, x2, y2, sc), axis=1)
    rectangles = rect2square(rectangles)
    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0,      rectangles[i][0]))
        y1 = int(max(0,      rectangles[i][1]))
        x2 = int(min(width,  rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        sc = rectangles[i][4]
        if x2 > x1 and y2 > y1:
            pick.append([x1, y1, x2, y2, sc])
    return NMS(pick, iou_threshold)


def filter_face_48net(cls_prob, roi, pts, rectangles, width, height, score_threshold, iou_threshold):
    """
    Function:
        Filter face position and calibrate bounding box on 12net's output
    Input:
        cls_prob  : cls_prob[1] is face possibility
        roi       : roi offset
        pts       : 5 landmark
        rectangles: 12net's predict, rectangles[i][0:3] is the position, rectangles[i][4] is score
        width     : image's origin width
        height    : image's origin height
        threshold : 0.7 can have 94% recall rate on CelebA-database
    Output:
        rectangles: face positions and landmarks
    """
    prob = cls_prob[:, 1]
    pick = np.where(prob >= score_threshold)
    rectangles = np.array(rectangles)
    x1 = rectangles[pick, 0]
    y1 = rectangles[pick, 1]
    x2 = rectangles[pick, 2]
    y2 = rectangles[pick, 3]
    sc = np.array([prob[pick]]).T

    dx1 = roi[pick, 0]
    dx2 = roi[pick, 1]
    dx3 = roi[pick, 2]
    dx4 = roi[pick, 3]

    w = x2-x1
    h = y2-y1

    pts0 = np.array([(w*pts[pick, 0]+x1)[0]]).T
    pts1 = np.array([(h*pts[pick, 5]+y1)[0]]).T
    pts2 = np.array([(w*pts[pick, 1]+x1)[0]]).T
    pts3 = np.array([(h*pts[pick, 6]+y1)[0]]).T
    pts4 = np.array([(w*pts[pick, 2]+x1)[0]]).T
    pts5 = np.array([(h*pts[pick, 7]+y1)[0]]).T
    pts6 = np.array([(w*pts[pick, 3]+x1)[0]]).T
    pts7 = np.array([(h*pts[pick, 8]+y1)[0]]).T
    pts8 = np.array([(w*pts[pick, 4]+x1)[0]]).T
    pts9 = np.array([(h*pts[pick, 9]+y1)[0]]).T
    x1 = np.array([(x1+dx1*w)[0]]).T
    y1 = np.array([(y1+dx2*h)[0]]).T
    x2  = np.array([(x2+dx3*w)[0]]).T
    y2 = np.array([(y2+dx4*h)[0]]).T
    rectangles = np.concatenate((x1, y1, x2, y2, sc, pts0, pts1, pts2, pts3, pts4, pts5, pts6, pts7, pts8, pts9),
                                axis=1)
    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0,      rectangles[i][0]))
        y1 = int(max(0,      rectangles[i][1]))
        x2 = int(min(width,  rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        if x2 > x1 and y2 > y1:
            pick.append([x1, y1, x2, y2, *rectangles[i][4:]])
    return NMS(pick, iou_threshold, use_iom=True)


def calculate_scales(img):
    """
    Function:
        calculate multi-scale and limit the maxinum side to 1000
    Input:
        img: original image
    Output:
        pr_scale: limit the maxinum side to 1000, < 1.0
        scales  : Multi-scale
    """
    pr_scale = 1.0
    h, w, _ = img.shape
    if min(w, h) > 1000:
        pr_scale = 1000.0 / min(h, w)
        w = int(w*pr_scale)
        h = int(h*pr_scale)
    elif max(w, h) < 1000:
        w = int(w*pr_scale)
        h = int(h*pr_scale)

    #multi-scale
    scales = []
    factor = 0.709
    factor_count = 0
    minl = min(h, w)
    while minl >= 12:
        scales.append(pr_scale*pow(factor, factor_count))
        minl *= factor
        factor_count += 1
    return scales
