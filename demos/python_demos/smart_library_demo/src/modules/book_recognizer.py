import cv2
import pyzbar.pyzbar as pyzbar
import numpy as np
from abc import ABC, abstractmethod


class BookRecognizer(ABC):
    @staticmethod
    def create(args):
        if args['name'] == 'QR':
            return QRBookRecognizer()
        
    @abstractmethod
    def recognize(self, frame):
        """Recognize book"""
        

class QRBookRecognizer(BookRecognizer):
    # Constructor
    def __init__(self):
        self.objects = []

    def recognize(self, frame):
        # Find barcodes and QR codes
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.objects = pyzbar.decode(gray)

        # Analise results
        ans = ""
        for obj in self.objects:
            if obj.type == 'QRCODE':
                ans = obj.data.decode('utf-8')

        # Return decode information
        return ans
