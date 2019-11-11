import sys, os
import numpy as np
import cv2 as cv
import ctypes as C

from abc import ABC, abstractmethod

refLandmarks = np.float32([[0.31556875000000000, 0.4615741071428571],  # left eye
                           [0.68262291666666670, 0.4615741071428571],  # right eye
                           [0.50026249999999990, 0.6405053571428571],  # tip of nose
                           [0.34947187500000004, 0.8246919642857142],  # left lip corner, right lip corner
                           [0.65343645833333330, 0.8246919642857142]]) # right lip corner


class FaceRecognizer(ABC):
     @staticmethod
     def create(args): 
        if args['name'] == 'DNNfr':
            return DNNRecognizer(args['rdXML'],
                   args['rdWidth'], args['rdHeight'], args['rdThreshold'], 
                   args['fdName'], args['fdXML'], 
                   args['fdWidth'], args['fdHeight'], args['fdThreshold'],
                   args['lmName'], args['lmXML'],
                   args['lmWidth'], args['lmHeight'])
        else:
            raise Exception('Error: wrong recognizer name')

     @abstractmethod
     def register(self, img, ID):
         '''Register new reader'''

     @abstractmethod
     def recognize(self, img):
         '''Recognize valid user'''

class FaceDetector(ABC):
     @staticmethod
     def create(args):
         if args['name'] == 'DNNfd':
            return DNNDetector(args['modelXML'], args['width'],
                              args['height'], args['threshold'])
         else:
            raise Exception('Error: wrong detector name')

     @abstractmethod
     def detect(self, img,  threshold):
         '''Detect faces on image'''

class FaceLandmarks(ABC):
    @staticmethod
    def create(args): 
         if args['name'] == 'DNNLandmarks':
            return DNNLandmarks(args['modelXML'], args['width'],
                                               args['height'])
         else:
            raise Exception('Error: wrong detector name')

    @abstractmethod
    def align(self, img, l):
         '''Detect faces on image'''

class DNNLandmarks(ABC):
    def __init__(self, modelXML, width, height):
        self.modelXML = modelXML
        self.modelBIN =  os.path.splitext(self.modelXML)[0] + '.bin'
        self.width = width
        self.height = height
        backendId = cv.dnn.DNN_BACKEND_INFERENCE_ENGINE
        targetId = cv.dnn.DNN_TARGET_CPU
        self.net = cv.dnn.readNet(self.modelBIN, self.modelXML)
        self.net.setPreferableBackend(backendId)
        self.net.setPreferableTarget(targetId)

    def findLandmarks(self, img):
        try:    
            blob = cv.dnn.blobFromImage(img,  size=(self.width, self.height))
            self.net.setInput(blob)
            out = self.net.forward()
            out = out.flatten()
            landmarks = np.empty((5, 2), dtype=np.float32)
            for i in range(5):
                landmarks[i] = [out[2*i],out[2*i+1] ]
            return landmarks
        except Exception as e:
            print('exception: ' + str(e))
            return np.zeros((5, 2), dtype=np.float32)

    def getTransform(self, src, dst):
        col_mean_src = cv.reduce(src, 0, cv.REDUCE_AVG)
        for row in src:
            row-=col_mean_src[0]

        col_mean_dst = cv.reduce(dst, 0, cv.REDUCE_AVG)
        for row in dst:
            row-=col_mean_dst[0]

        mean, dev_src = cv.meanStdDev(src)
        dev_src[0,0] = max(sys.float_info.epsilon, dev_src[0])
        src /= dev_src[0,0]
     
        mean, dev_dst = cv.meanStdDev(dst)
        dev_dst[0,0] = max(sys.float_info.epsilon, dev_dst[0])
        dst /= dev_dst[0,0] 
        
        w, u, vt = cv.SVDecomp(np.dot(cv.transpose(src), dst))
        r = cv.transpose(np.dot(u,vt))
        m = np.empty((2, 3), dtype=np.float32)
        m[0:2,0:2] = np.dot(r , (dev_dst[0,0] / dev_src[0,0]))
        m[0:2,2:3] = cv.transpose(col_mean_dst) - np.dot(m[0:2,0:2], 
                                                   cv.transpose(col_mean_src))
        return m


    def align(self, img, landmarks, refLandmarks):
        aligned_face = np.copy(img)
        refLandmarksCopy =  np.copy(refLandmarks)
        for  point, refPoint in zip(landmarks, refLandmarksCopy):
           point[1] = int(point[1]*img.shape[0])
           point[0] = int(point[0]*img.shape[1])
           refPoint[1] = int(refPoint[1]*img.shape[0])
           refPoint[0] = int(refPoint[0]*img.shape[1])
        m = self.getTransform(landmarks, refLandmarksCopy)
        aligned_face = cv.warpAffine(aligned_face, m, 
                            (aligned_face.shape[1], aligned_face.shape[0])) 
        return aligned_face
       
class DNNDetector(FaceDetector):
    def __init__(self, modelXML, width, height, threshold):
        self.modelXML = modelXML
        self.modelBIN = os.path.splitext(self.modelXML)[0] + '.bin'
        self.width = width
        self.height = height
        self.threshold = threshold
        backendId = cv.dnn.DNN_BACKEND_INFERENCE_ENGINE
        targetId = cv.dnn.DNN_TARGET_CPU
        self.net = cv.dnn.readNet(self.modelBIN, self.modelXML)
        self.net.setPreferableBackend(backendId)
        self.net.setPreferableTarget(targetId)

    def detect(self, img):
        blob = cv.dnn.blobFromImage(img,  size=(self.width, self.height))
        self.net.setInput(blob)
        out	= self.net.forward()
        faces = []
        for detection in out.reshape(-1, 7):
            confidence = float(detection[2])
            if confidence >  self.threshold:
                xmin = int(detection[3] *  img.shape[1]) if int(detection[3] *  img.shape[1]) > 0 else 0
                ymin = int(detection[4] *  img.shape[0]) if int(detection[4] *  img.shape[0]) > 0 else 0
                xmax = int(detection[5] *  img.shape[1]) if int(detection[5] *  img.shape[1]) > 0 else 0
                ymax = int(detection[6] *  img.shape[0]) if int(detection[6] *  img.shape[0]) > 0 else 0
                faces.append(((xmin, ymin), (xmax, ymax)))
        return faces

class DNNRecognizer(FaceRecognizer):
    def __init__(self, recXML, recWidth, recHeight, recThreshold, 
                  detName, detXML, detWidth, detHeight, detThreshold,
                  lmarksName, lmarksXML, lmarksWidth, lmarksHeight):
        args = dict(name = lmarksName, modelXML = lmarksXML,
                    width = lmarksWidth, height = lmarksHeight)
        self.fl = FaceLandmarks.create(args)
        args = dict(name = detName, modelXML = detXML,
                    width = detWidth, height = detHeight, threshold = detThreshold)
        self.det = FaceDetector.create(args)

        self.bd = np.empty((0, 256), dtype=np.float32)
        self.counter = 0 
        self.modelXML = recXML
        self.modelBIN = os.path.splitext(self.modelXML)[0] + '.bin'
        self.width = recWidth
        self.height = recHeight
        self.threshold = recThreshold
        backendId = cv.dnn.DNN_BACKEND_INFERENCE_ENGINE
        targetId = cv.dnn.DNN_TARGET_CPU
        self.net = cv.dnn.readNet(self.modelBIN, self.modelXML)
        self.net.setPreferableBackend(backendId)
        self.net.setPreferableTarget(targetId)

    def similarity(self, fVec, refVecs):
        refVecs =  refVecs.T  
        if fVec.size and  refVecs.size:
          return np.dot(fVec, refVecs)/(np.linalg.norm(fVec)*np.linalg.norm(refVecs, axis=0))
        else:
          return np.zeros((1, 1))

    def getFeatures(self, img):
        faces = self.det.detect(img)
        if len(faces) == 1:
            face = faces[0]
            roi = img[face[0][1]:face[1][1], face[0][0]:face[1][0]]
            landmarks = self.fl.findLandmarks(roi)
            alignFace = self.fl.align(roi, landmarks, refLandmarks)
            blob = cv.dnn.blobFromImage(alignFace,  size=(self.width, self.height))
            self.net.setInput(blob)
            out	= self.net.forward()
            featureVec = out.flatten()
        else:
            featureVec = np.empty(0)
        return (faces, featureVec)

    def recognize(self, img):
        faces, fVec = self.getFeatures(img)
        return (faces, self.similarity(fVec, self.bd))
    
    def register(self, img, ID = 0):
        _, vec = self.getFeatures(img)
        self.bd = np.append(self.bd, [vec], axis=0)
        self.counter = self.bd.shape[0]
        return self.counter
    
    

