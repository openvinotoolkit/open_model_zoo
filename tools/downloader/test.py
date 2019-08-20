import os

#
# Test paths from MO args
#
from topologies import ssd300

t = ssd300()
xmlPath, binPath = t.getIR()
xmlPath_fp16, binPath_fp16 = t.getIR(precision='FP16')

for path, name in zip ([t.model, t.config, xmlPath, binPath, xmlPath_fp16, binPath_fp16],
                       ['VGG_VOC0712Plus_SSD_300x300_ft_iter_160000.caffemodel',
                        'deploy.prototxt',
                        'ssd300.xml', 'ssd300.bin',
                        'ssd300.xml', 'ssd300.bin']):
    assert(os.path.exists(path)), path
    assert(os.path.basename(path) == name), name
    os.remove(path)

#
# Check Intel models
#
from topologies import vehicle_license_plate_detection_barrier_0106

t = vehicle_license_plate_detection_barrier_0106('FP16')
xmlPath, binPath = t.config, t.model
xmlPathIR, binPathIR = t.getIR()

for path, ref in zip([xmlPath, binPath, xmlPathIR, binPathIR],
                     ['FP16/vehicle-license-plate-detection-barrier-0106.xml',
                      'FP16/vehicle-license-plate-detection-barrier-0106.bin',
                      'FP16/vehicle-license-plate-detection-barrier-0106.xml',
                      'FP16/vehicle-license-plate-detection-barrier-0106.bin']):
    assert(os.path.exists(path)), path
    assert(path.endswith(ref)), ref

net = t.getIENetwork()

os.remove(xmlPath)
os.remove(binPath)

#
# Test inference with OpenCV
#
import cv2 as cv
from topologies import mobilenet_ssd

def iou(a, b):
    def inter_area(box1, box2):
        x_min, x_max = max(box1[0], box2[0]), min(box1[2], box2[2])
        y_min, y_max = max(box1[1], box2[1]), min(box1[3], box2[3])
        return (x_max - x_min) * (y_max - y_min)

    def area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    inter = inter_area(a, b)
    return inter / (area(a) + area(b) - inter)


t = mobilenet_ssd()
try:
    net = t.getOCVModel()
except:
    net = t.getOCVModel(useIR=False)


img = cv.imread(os.path.join(os.environ['OPENCV_TEST_DATA_PATH'], 'dnn', 'dog416.png'))
assert(not img is None), 'Test image'

classIds, confidences, boxes = net.detect(img, confThreshold=0.5)
refClassIds = [2, 7, 12]
refConfs = [0.99348623, 0.99497199, 0.99050343]
refBoxes = [[56, 102, 256, 204], [252, 53, 120, 71], [74, 150, 104, 242]]

assert(len(classIds) == len(refClassIds))
assert(len(confidences) == len(refConfs))
assert(len(boxes) == len(refBoxes))

iouThr = 1e-4
confThr = 1e-5

for refCl, refConf, refBox in zip(refClassIds, refConfs, refBoxes):
    matched = False
    for cl, conf, box in zip(classIds, confidences, boxes):
        if 1.0 - iou(box, refBox) < iouThr and cl == refCl and abs(conf - refConf) < confThr:
            matched = True
    assert(matched), 'Class %d' % refCl

os.remove(t.model)
os.remove(t.config)

#
# Test aliases
#
from topologies import text_detection_0004

t = text_detection_0004('FP16')
xmlPath, binPath = t.config, t.model

for path, ref in zip([t.config, t.model],
                     ['FP16/text-detection-0004.xml',
                      'FP16/text-detection-0004.bin',]):
    assert(os.path.exists(path)), path
    assert(path.endswith(ref)), ref

os.remove(t.config)
os.remove(t.model)
